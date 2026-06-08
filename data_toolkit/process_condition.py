# import os
# os.environ.setdefault("LIDRA_SKIP_INIT", "1")

# from pathlib import Path
# import torch
# from src.utils.train_utils import get_configs
# from src.train import ConditionEmbedding, init_ss_condition_embedder
# from src.datasets import Future3DDataset
# import tqdm 

# def main():
#     repo_root = Path(__file__).resolve().parents[1]
#     cfg_path = repo_root / "configs" / "mp8_nt512.yaml"  # adjust if needed
#     configs = get_configs(str(cfg_path))
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Initialize the condition embedder and wrapper using the same mapping as train.py
#     ss_cond_emb = init_ss_condition_embedder(
#         ss_generator_config_path=os.path.join("checkpoints", "hf", "ss_generator.yaml"),
#         ss_generator_ckpt_path=os.path.join("checkpoints", "hf", "ss_generator.ckpt"),
#         workspace_dir=str(repo_root),
#         device=device,
#     )
#     ss_condition_embedding = ConditionEmbedding(
#         ss_cond_emb, configs.get("ss_condition_input_mapping", [])
#     )

#     # Choose split (training=False for test split)
#     dataset = Future3DDataset(configs=configs, training=True)

#     out_dir = repo_root / "condition_outputs"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     with torch.no_grad():
#         for idx in tqdm.tqdm(range(len(dataset))):
#             sample = dataset[idx]
#             # Move tensors to device and cast to float
#             for k, v in sample.items():
#                 if isinstance(v, torch.Tensor):
#                     sample[k] = v.to(device).float()

#             cond_args, _ = ss_condition_embedding(sample)
#             cond = cond_args[0] if len(cond_args) > 0 else None
#             uid = sample.get("uid", f"sample_{idx}")
#             torch.save(cond.cpu(), out_dir / f"{uid}.pt")
#             print(f"[{idx+1}/{len(dataset)}] saved {uid}.pt, shape={tuple(cond.shape)}")


# if __name__ == "__main__":
#     main()
import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

from pathlib import Path
import torch
import torch.distributed as dist
from src.utils.train_utils import get_configs
from src.train import ConditionEmbedding, init_ss_condition_embedder
from src.datasets import Future3DDataset, MultiEpochsDataLoader
import tqdm


def ddp_setup():
    """
    torchrun으로 실행될 때 환경변수로 rank/world_size/local_rank가 들어옵니다.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # 단일 프로세스 실행 fallback
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def move_to_device(sample: dict, device: torch.device) -> dict:
    """
    uid(str) 같은 메타는 그대로 두고, Tensor만 device로 이동.
    dtype은 보존하되, float64는 float32로 다운캐스트(선택적으로).
    """
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            t = v.to(device, non_blocking=True)
            # 필요 시 float64 -> float32 (모델이 float32 기대하는 경우가 많음)
            if t.dtype == torch.float64:
                t = t.float()
            out[k] = t
        else:
            out[k] = v
    return out


def main():
    rank, world_size, local_rank = ddp_setup()

    split = "train"
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "configs" / "mp8_nt512.yaml"
    configs = get_configs(str(cfg_path))

    # 각 프로세스는 자신의 GPU를 사용
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # rank별로 출력 폴더는 동일해도 OK (idx shard로 충돌 방지)
    out_dir = Path("/mnt/storage/jhkim/3D-FUTURE/condition_outputs" if split=="train" else "/mnt/storage/jhkim/3D-FUTURE/condition_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Condition embedder init (각 프로세스에서 1회)
    ss_cond_emb = init_ss_condition_embedder(
        ss_generator_config_path=os.path.join("checkpoints", "hf", "ss_generator.yaml"),
        ss_generator_ckpt_path=os.path.join("checkpoints", "hf", "ss_generator.ckpt"),
        workspace_dir=str(repo_root),
        device=str(device),
    )
    ss_condition_embedding = ConditionEmbedding(
        ss_cond_emb, configs.get("ss_condition_input_mapping", [])
    )

    dataset = Future3DDataset(configs=configs, training=split=="train", use_latent=False)
    # data_loader = MultiEpochsDataLoader(
    #     dataset,
    #     batch_size=1,
    #     num_workers=8,
    #     drop_last=True,
    #     pin_memory=True,
    # )
    n = len(dataset)

    # rank마다 서로 다른 인덱스만 처리: rank, rank+world_size, ...
    indices = range(rank, n, world_size)[5000:]

    # tqdm는 rank0만 표시 (로그 난잡 방지)
    iterator = indices
    if rank == 0:
        iterator = tqdm.tqdm(list(indices), desc=f"rank{rank}/{world_size}", total=(n + world_size - 1) // world_size)

    torch.set_grad_enabled(False)

    processed = 0
    for idx in iterator:
        
        sample = dataset[idx]
        uid = sample.get("uid", f"sample_{idx}")
        # uid가 혹시 Tensor로 들어오면 str로 변환
        if isinstance(uid, torch.Tensor):
            uid = uid.item()
        uid = str(uid)
        save_path = out_dir / f"{uid}.pt"

        if os.path.exists(save_path):
            print(f"[idx={idx}] exist {uid}.pt")
            continue
        sample = move_to_device(sample, device)
        

        with torch.no_grad():
            cond_args, _ = ss_condition_embedding(sample)
        cond = cond_args[0] if (cond_args is not None and len(cond_args) > 0) else None

        if cond is None:
            # 필요하면 스킵 파일을 남기거나 로그만 남기십시오.
            # 여기서는 스킵 로그만.
            if rank == 0:
                print(f"[idx={idx}] cond is None -> skipped (uid={uid})")
            continue

        
        torch.save(cond.detach().cpu(), save_path)
        processed += 1

        # rank0만 상세 로그 (원하면 모든 rank 출력 가능)
        if rank == 0:
            print(f"[idx={idx}] saved {uid}.pt, shape={tuple(cond.shape)}")

    # DDP 종료
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"Done. world_size={world_size}, dataset={n}, processed(rank0)={processed}")


if __name__ == "__main__":
    main()

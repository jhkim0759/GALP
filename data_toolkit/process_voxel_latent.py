import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import json
import glob
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from src.sam3d_objects.model.backbone.tdfy_dit.models.sparse_structure_vae import (
    SparseStructureEncoderTdfyWrapper,
    SparseStructureDecoderTdfyWrapper,
)


def load_trellis_ss_wrapper(base_path, kind, device):
    base = Path(base_path)
    json_path = base.with_suffix(".json")
    ckpt_path = base.with_suffix(".safetensors")
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    with open(json_path) as f:
        cfg = json.load(f)
    args = cfg.get("args", {})

    if kind == "encoder":
        model = SparseStructureEncoderTdfyWrapper(
            **args,
            sample_posterior=False,
            return_raw=True,
            pretrained_ckpt_path=str(ckpt_path),
        )
    else:
        model = SparseStructureDecoderTdfyWrapper(
            **args,
            pretrained_ckpt_path=str(ckpt_path),
            return_raw=True,
        )

    model = model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def shard_list(items, rank, world_size):
    # rank별로 균등 분배 (round-robin)
    return items[rank::world_size]


def worker(rank, world_size, model_paths, trellis_encoder_ckpt, resolution=32):
    # GPU 8개, GPU당 4프로세스 => rank(0~31) -> gpu_id(0~7)
    gpu_id = rank // 8
    device = f"cuda:{gpu_id}"

    torch.cuda.set_device(gpu_id)

    # 프로세스당 1회만 모델 로드
    ss_encoder =  load_trellis_ss_wrapper(
                trellis_encoder_ckpt,
                kind="encoder",
                device=device,
            )

    my_paths = shard_list(model_paths, rank, world_size)

    for model_path in my_paths:
        voxel_path = os.path.join(model_path, f"voxel_{resolution}.npy")
        out_path = os.path.join(model_path, f"voxel_latent_{resolution}.pt")
        if not os.path.exists(voxel_path):
            print(f"[rank {rank} | {device}] skip (no voxel.npy): {model_path}")
            continue
        if os.path.exists(out_path):
            print(f"[rank {rank} | {device}] exist (voxel latent.npy): {model_path}")
            continue

        try:
            voxel = np.load(voxel_path)
            voxel_tensor = torch.from_numpy(voxel).to(device).unsqueeze(0)

            with torch.no_grad():
                # return_raw=True 필요하면 유지
                latents = ss_encoder(voxel_tensor)

            latents = {"mean": latents['mean'].cpu(), "logvar": latents['logvar'].cpu()}

            
            torch.save(latents, out_path)

            print(f"[rank {rank} | {device}] saved: {out_path}")

        except Exception as e:
            print(f"[rank {rank} | {device}] ERROR on {model_path}: {e}")


def main():
    # 처리 대상 경로 수집
    model_paths = sorted(glob.glob("/mnt/storage/jhkim/3D-FUTURE/3D-FUTURE-model/*"))

    trellis_encoder_ckpt = os.path.join(
        "checkpoints", "ckpt/ckpts/ss_enc_conv3d_16l8_fp16"
    )

    # GPU 8개 * GPU당 4프로세스
    world_size = 1  # 32
    resolution = 16

    # CUDA 멀티프로세스는 spawn 권장
    mp.set_start_method("spawn", force=True)

    mp.spawn(
        worker,
        args=(world_size, model_paths, trellis_encoder_ckpt, resolution),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()

import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

from pathlib import Path
import glob
import tqdm
import numpy as np
from PIL import Image
from collections import namedtuple

import torch
import torch.distributed as dist

from moge.model.v1 import MoGeModel


# -------------------------
# (1) Minimal pytorch3d replacement (your code kept)
# -------------------------
from pytorch3d.renderer import look_at_view_transform  # type: ignore
from pytorch3d.transforms import Transform3d  # type: ignore

DecomposedTransform = namedtuple("DecomposedTransform", ["scale", "rotation", "translation"])


def camera_to_pytorch3d_camera(device="cpu") -> DecomposedTransform:
    r3_to_p3d_R, r3_to_p3d_T = look_at_view_transform(
        eye=np.array([[0, 0, -1]]),
        at=np.array([[0, 0, 0]]),
        up=np.array([[0, -1, 0]]),
        device=device,
    )
    return DecomposedTransform(
        rotation=r3_to_p3d_R,
        translation=r3_to_p3d_T,
        scale=torch.tensor(1.0, dtype=r3_to_p3d_R.dtype, device=device),
    )


# -------------------------
# (2) DDP helpers
# -------------------------
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def is_main_process(rank: int) -> bool:
    return rank == 0


# -------------------------
# (3) Main
# -------------------------
def main():
    rank, world_size, local_rank = ddp_setup()

    # GPU binding per process
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    # Load model per process (each rank on its own GPU)
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    moge_model.eval()

    # Paths
    split = "train"  # "train" or "test"
    out_root = "/mnt/storage/jhkim/coco/val2017"
    out_dir_name = "moge_pointmap" if split == "train" else "moge_pointmap_test"
    out_dir = os.path.join(out_root, out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Build file list once and shard by rank
    pattern = os.path.join(out_root, f"images/*.jpg")
    image_paths = sorted(glob.glob(pattern))
    n = len(image_paths)

    # Shard: rank, rank+world_size, ...
    my_paths = image_paths[rank::world_size]

    iterator = my_paths
    if is_main_process(rank):
        iterator = tqdm.tqdm(my_paths, desc=f"rank{rank}/{world_size} (images)", total=len(my_paths))

    # Precompute camera transform once per process
    camera_convention_transform = (
        Transform3d()
        .rotate(camera_to_pytorch3d_camera(device=device).rotation)
        .to(device)
    )

    for image_path in iterator:
        uid = os.path.basename(image_path).split(".")[0]
        save_path = os.path.join(out_dir, f"{uid}.npy")

        # (Optional) resume-safe: skip if already exists
        # if os.path.exists(save_path):
        #     continue

        # Load image -> tensor
        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            moge_output = moge_model.infer(image_tensor)

        # Points transform
        pointmaps = moge_output["points"]  # Tensor
        pt3d_points = camera_convention_transform.transform_points(pointmaps)
        moge_output["pt3d_points"] = pt3d_points

        # Move tensors -> numpy for saving
        save_dict = {}
        for k, v in moge_output.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.detach().cpu().numpy()
            else:
                # non-tensor outputs (if any) -> store as numpy object
                save_dict[k] = np.array(v, dtype=object)

        # Save as npz (safe dict-like)
        np.save(save_path, save_dict)

        if is_main_process(rank):
            shape_str = tuple(save_dict["pt3d_points"].shape) if "pt3d_points" in save_dict else None
            print(f"saved {save_path} pt3d_points={shape_str}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if is_main_process(rank):
        print(f"Done. total_images={n}, world_size={world_size}")


if __name__ == "__main__":
    main()

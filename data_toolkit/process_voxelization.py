import os
os.environ.setdefault("LIDRA_SKIP_INIT", "1")

import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.datasets.data_utils import mesh_to_voxel_tensor


def process_one(model_path: str, resolution: int = 32, sample_points: bool = True):
    voxel_path = os.path.join(model_path, f"voxel_{resolution}.npy")
    points_path = os.path.join(model_path, "points.npy")
    mesh_path = os.path.join(model_path, "normalized_model.obj")

    # 이미 결과가 있으면 스킵
    if os.path.exists(voxel_path) and os.path.exists(points_path):
        return ("skip", model_path, None)

    if not os.path.exists(mesh_path):
        return ("missing_mesh", model_path, None)

    voxel_tensor, points = mesh_to_voxel_tensor(
        mesh_path,
        resolution=resolution,
        sample_points=sample_points,
    )

    voxel = np.array(voxel_tensor)
    points = np.array(points)

    # 더 안전하게: 임시 저장 후 원자적 교체
    np.save(voxel_path, voxel)
    np.save(points_path, points)

    return ("ok", model_path, None)


def main():
    model_paths = sorted(glob.glob("/mnt/storage/jhkim/3D-FUTURE/3D-FUTURE-model/*"))

    # CPU 프로세스 수: 필요 시 예) max_workers=32 같이 고정 가능
    max_workers = max(16, os.cpu_count() or 1)

    stats = {"ok": 0, "skip": 0, "missing_mesh": 0, "error": 0}

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_one, p, 16, True) for p in model_paths]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="voxelize"):
            try:
                status, model_path, _ = fut.result()
                stats[status] += 1
                if status == "missing_mesh":
                    # 필요하면 출력
                    # print("missing mesh:", model_path)
                    pass
            except Exception as e:
                stats["error"] += 1
                # 어떤 경로에서 터졌는지 보려면 아래처럼 바꿔서 submit 시 path를 묶어도 됩니다.
                print("ERROR:", repr(e))

    print("Done:", stats)


if __name__ == "__main__":
    main()

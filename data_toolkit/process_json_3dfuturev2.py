import os
import math
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor

from scipy.spatial.transform import Rotation as R

def rotmat_to_6d(rot: np.ndarray) -> np.ndarray:
    """
    R: (..., 3, 3) shape의 회전행렬
    return: (..., 6) shape의 6D rotation 표현
    """
    assert rot.shape[-2:] == (3, 3), "입력 R의 shape는 (..., 3, 3) 이어야 합니다."
    
    r1 = rot[..., :, 0]  # 첫 번째 column, shape (..., 3)
    r2 = rot[..., :, 1]  # 두 번째 column, shape (..., 3)
    
    # 마지막 차원을 기준으로 이어붙이기 → (..., 6)
    rot_6d = np.concatenate([r1, r2], axis=-1)
    return rot_6d


def rot6d_to_rotmat(rot_6d: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    rot_6d: (..., 6) shape의 6D rotation 표현
    return: (..., 3, 3) 회전행렬
    """
    assert rot_6d.shape[-1] == 6, "입력 rot_6d의 마지막 차원은 6이어야 합니다."
    
    # a1, a2를 분리: (..., 3)
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    
    # r1 = normalize(a1)
    r1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + eps)
    
    # a2에서 r1 성분 제거: a2_ortho = a2 - (r1·a2) r1
    dot = np.sum(r1 * a2, axis=-1, keepdims=True)  # (..., 1)
    a2_ortho = a2 - dot * r1
    
    # r2 = normalize(a2_ortho)
    r2 = a2_ortho / (np.linalg.norm(a2_ortho, axis=-1, keepdims=True) + eps)
    
    # r3 = r1 × r2
    r3 = np.cross(r1, r2, axis=-1)
    
    # column 기준으로 쌓아서 (..., 3, 3) 만들기
    # 각 r*는 (..., 3) 이므로, 마지막 축 기준으로 쌓고 transpose
    R = np.stack([r1, r2, r3], axis=-1)  # (..., 3, 3)
    return R


def transform_from_local(local_pos, local_euler, ref_pos, ref_euler):
    # Convert euler angles to rotation matrices
    if len(ref_euler) == 4:
        ref_euler = R.from_quat(ref_euler, scalar_first=True).as_euler('xyz', degrees=True)
    if len(local_euler) == 4:
        local_euler = R.from_quat(local_euler, scalar_first=True).as_euler('xyz', degrees=True)
        
    rot_ref = R.from_euler('xyz', ref_euler, degrees=True).as_matrix()
    local_rot = R.from_euler('xyz', local_euler, degrees=True).as_matrix()
    
    # Compute global position
    global_pos = np.dot(rot_ref, local_pos) + np.array(ref_pos)
    
    # Compute global rotation
    global_rot = np.dot(rot_ref, local_rot)
    global_euler = R.from_matrix(global_rot).as_quat(scalar_first=True)
    
    return tuple(global_pos), tuple(global_euler)

def transform_to_query(pos_target, euler_target, pos_query, euler_query):
    rot_query = R.from_euler('xyz', euler_query, degrees=True).as_matrix()
    rot_target = R.from_euler('xyz', euler_target, degrees=True).as_matrix()

    relative_pos = np.dot(rot_query.T, np.array(pos_target) - np.array(pos_query))

    relative_rot = np.dot(rot_query.T, rot_target)
    relative_quat = R.from_matrix(relative_rot).as_quat(scalar_first=True)

    return relative_pos, relative_quat

# --------- globals set in initializer ----------
G_COCO = None
G_ASSET_DIR = None
G_SET = None


def _init_worker(scene_json_path, asset_dir, set_name):
    """
    각 프로세스에서 COCO를 1회 로드해 전역으로 보관.
    """
    global G_COCO, G_ASSET_DIR, G_SET
    G_COCO = COCO(scene_json_path)
    G_ASSET_DIR = asset_dir
    G_SET = set_name


def process_one_image(img_id_str):
    """
    img_id 하나(장면 1개)를 처리하여 data_dict 반환.
    실패/빈 결과는 None 반환.
    """
    global G_COCO, G_ASSET_DIR, G_SET

    img_id = int(img_id_str)

    data_dict = {
        "image_path": f"{G_SET}/image/{str(img_id).zfill(7)}.jpg",
        "translation": [],
        "rotation": [],
        "6d_rotation": [],
        "scale": [],
        "object_id": [],
    }

    ann_ids = G_COCO.getAnnIds(imgIds=img_id+1)
    annotations = G_COCO.loadAnns(ann_ids)

    model_ids = []
    translations = []
    eulers = []
    trimesh_scene = trimesh.Scene()
    camera_transform = np.eye(4)
    fov = math.radians(60)
    # try:
    scales = []
    centers = []
    
    for ann in annotations:
        model_id = ann['model_id']
        if model_id is None:
            continue
        if 'pose' not in ann or ann['pose']['translation'] is None:
            continue
        model_path = os.path.join(G_ASSET_DIR, model_id, 'raw_model.obj')
        if not os.path.exists(model_path):
            continue
        model = trimesh.load(model_path)
        
        vertices = model.vertices
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        vertices_centered = vertices - center
        scale = vertices_centered.max()
        
        model_ids.append(model_id)
        translations.append(ann['pose']['translation'])
        eulers.append(ann['pose']['euler'])
        scales.append(scale)
        centers.append(center)
        
        if 'fov' in ann:
            fov = ann['fov']
    
    if len(model_ids) == 0:
        return

    translations.append((0.0, 0.0, 0.0))
    eulers.append((0.0, 0.0, 0.0))
    
    quats = []
    poses = []
    for translation, euler in zip(translations, eulers):
        relative_pos, relative_quat = transform_to_query(translation, euler, translations[0], eulers[0])
        poses.append(relative_pos)
        quats.append(relative_quat)
    
    pos_query = (0.0, 0.0, 0.0)
    quat_query = R.from_euler('xyz', (90, 0, 0), degrees=True).as_quat(scalar_first=True)

    local_pos = []
    local_quat = []
    poses = poses[1:]  # Skip the first pose as it is the reference
    quats = quats[1:]  # Skip the first quaternion as it is the reference
    for pose, quat in zip(poses, quats):
        global_pos, global_quat = transform_from_local(pose, quat, pos_query, quat_query)
        local_pos.append(global_pos)
        local_quat.append(global_quat)
    positions = np.array([pos_query] + local_pos)
    quats = np.array([quat_query] + local_quat)

    for i in range(len(model_ids)):
        model_id = model_ids[i]
        transform = np.eye(4)
        rotation = R.from_quat(quats[i], scalar_first=True).as_matrix()
        
        data_dict["translation"].append(centers[i] @ rotation.T + positions[i])
        data_dict["rotation"].append(rotation)
        data_dict["scale"].append(scales[i])
        data_dict["object_id"].append(model_id)
        data_dict["6d_rotation"].append(rotmat_to_6d(rotation))

    data_dict["num_parts"] = len(data_dict["object_id"])
    return data_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="test", choices=["train", "val", "test"])
    parser.add_argument("--asset_dir", default="/mnt/storage/jhkim/3D-FUTURE/3D-FUTURE-model")
    parser.add_argument("--scene_root", default="/mnt/storage/jhkim/3D-FUTURE/3D-FUTURE-scene/GT")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    set_name = args.set
    scene_json_path = os.path.join(args.scene_root, f"{set_name}_set.json")
    out_path = args.out if args.out is not None else f"{set_name}_3dfuture.npy"

    # 메인 프로세스에서 img id 목록만 추출
    coco_main = COCO(scene_json_path)
    scene_ids = coco_main.getImgIds()
    scene_images = coco_main.loadImgs(scene_ids)
    image_ids = [str(scene_image["file_name"]) for scene_image in scene_images]

    dataset = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(scene_json_path, args.asset_dir, set_name),
    ) as ex:
        for result in tqdm(ex.map(process_one_image, image_ids), total=len(image_ids), leave=True):
            if result is not None:
                dataset.append(result)

    np.save(out_path, dataset, allow_pickle=True)
    print(f"Saved: {out_path}  (num_scenes={len(dataset)})")


if __name__ == "__main__":
    main()


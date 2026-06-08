import os
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')  # allow override for headless renderers

import numpy as np
import trimesh
import pyrender
import PIL
from PIL import Image
import torch
import torch.nn.functional as tf
from scipy.spatial.transform import Rotation as R   

def render_depth_from_mesh(
    mesh: trimesh.Trimesh,
    fov: float,
    image_size: tuple = (512, 512),
    camera_pose: np.ndarray = None
) -> np.ndarray:
    """
    주어진 fov(도 단위)와 카메라 pose로 mesh의 depth map만 렌더링.
    반환: (H, W) float32 또는 0~255 정규화 float32
    """
    if isinstance(mesh, trimesh.Trimesh):
        mesh_scene = trimesh.Scene(mesh)
    elif isinstance(mesh, trimesh.Scene):
        mesh_scene = mesh
    else:
        raise ValueError("mesh must be a trimesh.Trimesh or trimesh.Scene")

    # 1) trimesh.Scene -> pyrender.Scene
    scene = pyrender.Scene.from_trimesh_scene(mesh_scene)

    # 2) 카메라 설정 (fov 사용)
    H, W = image_size[1], image_size[0]
    camera = pyrender.PerspectiveCamera(
        yfov=fov,        # fov(도) -> 라디안
        aspectRatio=W / H
    )

    # 3) 카메라 pose (없으면 기본: Z+ 방향으로 radius=3.5 지점)
    if camera_pose is None:
        radius = 0
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, radius],
            [0.0, 0.0, 0.0, 1.0]
        ])

    cam_node = scene.add(camera, pose=camera_pose)

    # 4) 오프스크린 렌더러 생성
    renderer = pyrender.OffscreenRenderer(
        viewport_width=W,
        viewport_height=H
    )

    # 5) 렌더링 (image는 버리고 depth만 사용)
    color, depth = renderer.render(scene)

    # 6) 정리
    renderer.delete()
    scene.remove_node(cam_node)

    return depth.astype(np.float32)


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
ORIGIN = None

import glob 

def _init_worker(scene_json_path, asset_dir, set_name):
    """
    각 프로세스에서 COCO를 1회 로드해 전역으로 보관.
    """
    global G_COCO, G_ASSET_DIR, G_SET, ORIGIN
    G_COCO = COCO(scene_json_path)
    G_ASSET_DIR = asset_dir
    G_SET = set_name
    ORIGIN = os.path.dirname(G_ASSET_DIR)


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth_map.shape

    # Intrinsic parameters
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Generate grid of pixel coordinates
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # Unproject to camera coordinates
    x_cam = (v - cv) * depth_map / fv
    y_cam = (u - cu) * depth_map / fu  # -Z
    z_cam = depth_map # -Z

    # Stack to form camera coordinates
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords

def process_one_image(img_id_str):
    """
    img_id 하나(장면 1개)를 처리하여 data_dict 반환.
    실패/빈 결과는 None 반환.
    """
    global G_COCO, G_ASSET_DIR, G_SET, ORIGIN

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

    scales = []
    translations = []
    rotations = []
    centers = []
    
    scene = trimesh.Scene()
    h_size, w_size = 1200, 1200
    problem = "4eb55fc9-1a1a-4371-baf4-8f61f3ece58f"
    for ann in annotations:
        model_id = ann['model_id']
        if 'pose' not in ann or ann['pose']['translation'] is None:
            continue

        obj_file = os.path.join(G_ASSET_DIR, model_id, 'raw_model.obj')

        trans_vec = np.array(ann["pose"]['translation'], dtype=np.float32)
        rot_mat   = np.array(ann["pose"]['rotation'],    dtype=np.float32)

        mesh = trimesh.load(obj_file, force='mesh', skip_materials=True)
        if model_id==problem:
            mesh.vertices[:,0] *= -1 
        
        vertices = mesh.vertices
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
        vertices_centered = vertices - center
        scale = vertices_centered.max()
        mesh.vertices = vertices_centered/scale
        
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot_mat * scale
        transform[:3, 3]  = center @ rot_mat.T  + trans_vec 
        scene.add_geometry(mesh, transform=transform)

        scales.append(scale)
        rotations.append(rot_mat)
        translations.append(center @ rot_mat.T  + trans_vec)
        model_ids.append(model_id)

    if len(model_ids) == 0:
        print(f"{G_SET}/image/{str(img_id).zfill(7)}.jpg")
        return None    
    
    depth = render_depth_from_mesh(scene, fov=1.045, image_size=(h_size,w_size))
    masks = sorted(glob.glob(os.path.join(ORIGIN, "masked_images" if G_SET=='train' else "masked_images_test", str(img_id).zfill(7), "*mask.png")))
    image_mask = np.stack([np.array(PIL.Image.open(mask).convert("L")) for mask in masks])
    image_mask = image_mask.max(0)>0

    depth[depth==np.inf] = 0
    depth[~image_mask] = 0
    zero_mask = depth!=0

    moge = np.load(os.path.join(ORIGIN, "moge_pointmap" if G_SET=='train' else "moge_pointmap_test", f"{str(img_id).zfill(7)}.npy"), allow_pickle=True).item()
    moge_depth = np.array(moge['depth'].cpu(), dtype=np.float32) if isinstance(moge['depth'], torch.Tensor) else np.array(moge['depth'], dtype=np.float32)
    depth_scale = (moge_depth[zero_mask] / depth[zero_mask]).mean().item()
    
    for i in range(len(model_ids)):
        data_dict["translation"].append(translations[i]*depth_scale)
        data_dict["rotation"].append(rotations[i])
        data_dict["scale"].append(scales[i]*depth_scale)
        data_dict["object_id"].append(model_ids[i])
        data_dict["6d_rotation"].append(rotmat_to_6d(rotations[i]))
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

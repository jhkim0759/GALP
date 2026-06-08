import os
import math
import argparse
import numpy as np
import trimesh
from tqdm import tqdm
from pycocotools.coco import COCO
from concurrent.futures import ProcessPoolExecutor


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    roll, pitch, yaw: radians
    rotation order: ZYX (yaw → pitch → roll)
    """
    cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)

    R = np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy,     cy * sx,              cy * cx]
    ])

    return R


def rotation_matrix_to_6d(R):
    """
    R: (3,3) rotation matrix
    return: (6,) 6D rotation representation
    """
    # 첫 두 column 사용 (일반적인 정의)
    return R[:, :2].reshape(-1)


def euler_to_6d(euler_xyz):
    R = euler_to_rotation_matrix(euler_xyz[0], euler_xyz[1], euler_xyz[2])
    return rotation_matrix_to_6d(R)

# --------- math utils ----------
def euler_to_R(euler_xyz):
    T = trimesh.transformations.euler_matrix(
        euler_xyz[0], euler_xyz[1], euler_xyz[2],
        axes="sxyz"
    )
    return T[:3, :3]


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

    ann_ids = G_COCO.getAnnIds(imgIds=img_id)
    annotations = G_COCO.loadAnns(ann_ids)

    for ann in annotations:
        # 필수 키 검증
        model_id = ann.get("model_id", None)
        pose = ann.get("pose", None)
        if model_id is None or pose is None:
            continue
        if pose.get("translation", None) is None or pose.get("euler", None) is None:
            continue

        model_path = os.path.join(G_ASSET_DIR, model_id, "raw_model.obj")
        if not os.path.exists(model_path):
            continue

        # mesh load
        model = trimesh.load(model_path, force="mesh", skip_materials=True)
 
        t = np.array(pose["translation"], dtype=np.float32)
        euler = np.array(pose["euler"], dtype=np.float32)
        
        # normalize 파라미터(원 코드와 동일 로직 유지)
        c = model.bounding_box.centroid.astype(np.float32)
        s = float(np.max(model.bounding_box.extents) / 2.0) * 2.0

        R = euler_to_R(euler)
        t_adj = t + R @ c

        rot6d = euler_to_6d(euler)

        print(np.array(rot6d).shape)

        data_dict["translation"].append(t_adj.tolist())
        data_dict["rotation"].append(euler.tolist())
        data_dict["scale"].append(s)
        data_dict["object_id"].append(model_id)
        data_dict["6d_rotation"].append(rot6d)

    data_dict["num_parts"] = len(data_dict["object_id"])
    return data_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="test", choices=["train", "val", "test"])
    parser.add_argument("--asset_dir", default="../assets/3d_future/3D-FUTURE-model")
    parser.add_argument("--scene_root", default="../assets/3d_future/3D-FUTURE-scene/GT")
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
    image_ids = [str(scene_image["id"]) for scene_image in scene_images]

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


# from pycocotools.coco import COCO
# from tqdm import tqdm
# import trimesh 
# import numpy as np
# import os 
# import math
# def euler_to_R(euler_xyz):
#     # euler_xyz: [rx, ry, rz] (radian 가정)
#     # axes='sxyz'는 흔한 convention 중 하나입니다.
#     T = trimesh.transformations.euler_matrix(
#         euler_xyz[0], euler_xyz[1], euler_xyz[2],
#         axes='sxyz'
#     )
#     return T[:3, :3]

# set_ = "test"
# scene_json_path = f"../assets/3d_future/3D-FUTURE-scene/GT/{set_}_set.json"

# scene_data = COCO(scene_json_path)
# scene_ids = scene_data.getImgIds()
# scene_images = scene_data.loadImgs(scene_ids)
# image_ids = [str(scene_image['id']) for scene_image in scene_images]



# scene_mesh = trimesh.Scene()
# asset_dir = "../assets/3d_future/3D-FUTURE-model"
# dataset = []

# for img_id in tqdm(image_ids, leave=True):
#     img_info = scene_data.loadImgs(int(img_id))[0]
#     # Get the width and height of the image from img_info
#     width = img_info['width']
#     height = img_info['height']
    
#     models = []
#     translations = []
#     eulers = []
#     trimesh_scene = trimesh.Scene()
#     camera_transform = np.eye(4)
#     fov = math.radians(60)
#     # try:
#     ann_ids = scene_data.getAnnIds(imgIds=int(img_id))
#     annotations = scene_data.loadAnns(ann_ids)
    
#     data_dict = {"image_path": f"test/image/{str(img_id).zfill(7)}.jpg"}
#     data_dict["translation"] = []
#     data_dict["rotation"] = []
#     data_dict["scale"] = []
#     data_dict["object_id"] = []

#     for ann in annotations:
#         try:
#             model_id = ann['model_id']
#             if model_id is None:
#                 continue
#             if 'pose' not in ann or ann['pose']['translation'] is None:
#                 continue
#             model_path = os.path.join(asset_dir, model_id, 'raw_model.obj')
#             if not os.path.exists(model_path):
#                 continue
#             model = trimesh.load(model_path, skip_materials=True)
#         except:
#             continue
        
#         translations.append(ann['pose']['translation'])
#         eulers.append(ann['pose']['euler'])
#         if 'fov' in ann:
#             fov = ann['fov']


#         t = np.array(ann["pose"]["translation"], dtype=np.float32)
#         euler = np.array(ann["pose"]["euler"], dtype=np.float32)

#         c = model.bounding_box.centroid.astype(np.float32)  
#         s = (np.max(model.bounding_box.extents) / 2.0)
#         s = float(s)*2

#         # rotation matrix
#         R = euler_to_R(euler)

#         t_adj = t + R @ c

#         data_dict["translation"].append(t_adj.tolist())
#         data_dict["rotation"].append(euler.tolist())  # 회전은 그대로
#         data_dict["scale"].append(s)                  # s로 복원 스케일
#         data_dict["object_id"].append(ann["model_id"])

#     data_dict["num_parts"] = len(data_dict["object_id"])
#     dataset.append(data_dict)

# np.save(f"{set_}_3dfuture.npy", dataset)
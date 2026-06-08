import glob 
import tqdm
import os 
import trimesh

for mesh_path in tqdm.tqdm(glob.glob("../assets/3d_future/3D-FUTURE-model/*")):
    mesh_path = os.path.join(mesh_path,'raw_model.obj')
    save_path = mesh_path.replace('raw_model.obj','normalized_model.obj')
    mesh = trimesh.load(mesh_path, force="mesh")
    mesh.merge_vertices()
    vertices = mesh.vertices

    center = (vertices.min(0)+vertices.max(0))/2
    vertices_centered = vertices - center
    scale = vertices_centered.max()
    normalized = vertices_centered/scale
    mesh.vertices=normalized
    mesh.export(save_path)
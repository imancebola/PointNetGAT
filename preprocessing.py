"""
Mesh Preprocessing for PointNetGAT

This script preprocesses 3D mesh data (in .obj format) and corresponding 
per-vertex multiclass label files (.npy) for use with PointNetGAT using PyTorch Geometric.


Expected directory structure:
- One folder with .obj mesh files.
- One folder with matching .npy label files (same base names).

Output:
- Processed PyTorch Geometric datasets saved in:
    <output_dir>/<output_file>/
    ├── train/
    │   └── sample1.pt
    ├── val/
    │   └── sample2.pt
    └── test/
        └── sample3.pt
"""

import numpy as np
import torch
from torch_geometric.data import Data
import trimesh
import glob
import os
import re
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

def parse_obj_file(file_path, parse_edges=False):  # parse_edges = True for segmentations from MedMeshCNN
    vertices = []
    faces = []
    edges = []
    edge_labels = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                try:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (IndexError, ValueError):
                    print(f"Error in 'v' line of {file_path}: {line.strip()}")
            elif line.startswith('f '):
                parts = line.strip().split()
                try:
                    faces.append([int(p.split('/')[0]) - 1 for p in parts[1:]])
                except (IndexError, ValueError):
                    print(f"Error in 'f' line of {file_path}: {line.strip()}")
            elif parse_edges and line.startswith('e '):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        v1, v2, label = int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3])
                        edges.append([v1, v2])
                        edge_labels.append(label)
                    except ValueError as e:
                        print(f"Error in 'e' line of {file_path}: {line.strip()} - {e}")
                else:
                    print(f"Malformed 'e' line in {file_path}: {line.strip()}")
    edges = np.array(edges) if edges else np.empty((0, 2), dtype=np.int64)
    edge_labels = np.array(edge_labels) if edge_labels else np.empty(0, dtype=np.int64)
    return np.array(vertices), np.array(faces), edges, edge_labels

def faces_to_edges(faces):
    edges = set()
    for face in faces:
        for i in range(len(face)):
            v1, v2 = face[i], face[(i + 1) % len(face)]
            edges.add(tuple(sorted([v1, v2])))
    edges = np.array(list(edges)).T
    return edges

def normalize_vectors(vectors):
    norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
    return torch.where(norms > 0, vectors / (norms + 1e-8), vectors)

def preprocess_mesh_training(data_dir, mesh_file, multiclass_npy_file):
    vertices, faces, _, _ = parse_obj_file(os.path.join(data_dir, mesh_file), parse_edges=False)
    multiclass_labels = np.load(os.path.join(data_dir, multiclass_npy_file))

    if len(vertices) != len(multiclass_labels):
        raise ValueError(f"Mismatch in vertex count for {mesh_file}: "
                         f"OBJ={len(vertices)}, multiclass={len(multiclass_labels)}")

    vertices_tensor = torch.tensor(vertices, dtype=torch.float)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    centroid = torch.mean(vertices_tensor, dim=0)
    radial_vectors = vertices_tensor - centroid
    radial_vectors_normalized = normalize_vectors(radial_vectors)

    edge_index = faces_to_edges(faces)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    x = radial_vectors_normalized.numpy()

    data = Data(
        pos=vertices_tensor,
        x=torch.tensor(x, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(multiclass_labels, dtype=torch.long),
        faces=faces
    )
    return data, os.path.basename(mesh_file)

def load_train_data(data_dir, obj_file, labels_file, train, val, test):
    obj_dir = os.path.join(data_dir, obj_file)
    multiclass_dir = os.path.join(data_dir, labels_file)
    mesh_files = sorted(glob.glob(os.path.join(obj_dir, "*.obj")))

    def get_base_name(filename):
        return re.sub(r"\.mesh_12800\.obj$|_1500_labels\.npy$", "", os.path.basename(filename))

    mesh_dict = {get_base_name(f): f for f in mesh_files}
    multiclass_files = sorted(glob.glob(os.path.join(multiclass_dir, "*.npy")))
    multiclass_dict = {get_base_name(f): f for f in multiclass_files}

    common_keys = set(mesh_dict.keys()) & set(multiclass_dict.keys())

    data_list = []
    file_names = []
    all_labels = []
    for key in sorted(common_keys):
        mesh_file = mesh_dict[key]
        multi_file = multiclass_dict[key]
        data, file_name = preprocess_mesh_training(
            data_dir,
            os.path.relpath(mesh_file, data_dir),
            os.path.relpath(multi_file, data_dir)
        )
        data_list.append(data)
        file_names.append(file_name)
        all_labels.append(np.load(multi_file))

    all_labels = np.concatenate(all_labels)
    unique_labels = np.unique(all_labels)
    print(f"Unique classes in multiclass labels: {unique_labels}")
    if len(unique_labels) != 5 or not np.all(unique_labels == np.arange(5)):
        print("Warning: Multiclass labels do not contain exactly 5 classes (0 to 4)")

    train_size = train
    val_size = val
    test_size = test

    train_val_data, test_data, train_val_files, test_files = train_test_split(
        data_list, file_names, test_size=test_size, random_state=42, shuffle=True
    )

    relative_val_size = val_size / (train_size + val_size)
    train_data, val_data, train_files, val_files = train_test_split(
        train_val_data, train_val_files, test_size=relative_val_size, random_state=42, shuffle=True
    )

    print(f"Data split: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test")
    return train_data, val_data, test_data, train_files, val_files, test_files

def save_mesh_info(data_dir, train_data, val_data, test_data, train_files, val_files, test_files):
    output_file = os.path.join(data_dir, "processed_data", "mesh_info.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        f.write("Mesh information for each split\n\n")

        def write_set_info(set_name, data_list, file_list):
            f.write(f"=== {set_name} ({len(data_list)} meshes) ===\n")
            for data, file_name in zip(data_list, file_list):
                num_vertices = data.pos.shape[0]
                num_edges = data.edge_index.shape[1]
                num_faces = len(data.faces) if hasattr(data, 'faces') else 0
                f.write(f"Mesh: {file_name}\n")
                f.write(f"  Vertices: {num_vertices}\n")
                f.write(f"  Edges: {num_edges}\n")
                f.write(f"  Faces: {num_faces}\n\n")

        write_set_info("Training", train_data, train_files)
        write_set_info("Validation", val_data, val_files)
        write_set_info("Test", test_data, test_files)

    print(f"Mesh info saved at: {output_file}")

if __name__ == "__main__":
    data_dir = ""  # base path to the data
    output_dir = ""# base path to the output
    obj_file = ""  # folder name containing .obj mesh files
    labels_file = ""  # folder name containing .npy label files
    output_file = "preprocess_data"  # name for the output folder
    train_size = 0.8
    val_size = 0.16
    test_size = 0.14

    # Load data and filenames
    train_data, val_data, test_data, train_files, val_files, test_files = load_train_data(
        data_dir, obj_file, labels_file, train_size, val_size, test_size
    )

    # Save mesh statistics
    save_mesh_info(data_dir, train_data, val_data, test_data, train_files, val_files, test_files)

    # Save processed data into separate folders by original name
    base_output_dir = os.path.join(output_dir, output_file)
    for split_name, dataset, file_list in [
        ("train", train_data, train_files),
        ("val", val_data, val_files),
        ("test", test_data, test_files),
    ]:
        split_dir = os.path.join(base_output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for data, file_name in zip(dataset, file_list):
            base_name = os.path.splitext(os.path.basename(file_name))[0]
            output_path = os.path.join(split_dir, f"{base_name}.pt")
            torch.save(data, output_path)

    print(f"\nData saved to: {base_output_dir}")

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    # Inspect one batch from the train loader
    for batch in train_loader:
        print("\n=== Inspecting one batch from train_loader ===")
        print(f"Batch pos: {batch.pos.shape}")
        print(f"Batch x: {batch.x.shape}")
        print(f"Batch edge_index: {batch.edge_index.shape}")
        print(f"Batch y: {batch.y.shape}")
        print(f"Batch batch: {batch.batch.shape}")
        break

    print(f"\nLoaded data: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test")

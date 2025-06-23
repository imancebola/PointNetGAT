import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from torch_geometric.nn import PointNetConv, GATConv
from sklearn.metrics import accuracy_score
import trimesh

def load_data_from_folder(folder_path):
    pt_files = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
    print(f"Found {len(pt_files)} .pt files in {folder_path}")
    
    all_data = []
    basenames = []
    for pt_file in pt_files:
        try:
            data = torch.load(pt_file, weights_only=False)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                all_data.append(data)
            basenames.append(os.path.splitext(os.path.basename(pt_file))[0])
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
    return all_data, basenames

def intra_class_variance_loss(features, labels, num_classes=5):
    loss = 0
    for c in range(num_classes):
        class_features = features[labels == c]
        if class_features.size(0) > 1:
            mean_feature = class_features.mean(dim=0)
            variance = ((class_features - mean_feature) ** 2).mean()
            loss += variance
    return loss / num_classes

class PointNetGAT(nn.Module):
    def __init__(self, in_channels=3, out_channels=5):
        super(PointNetGAT, self).__init__()
        self.conv1 = PointNetConv(local_nn=nn.Sequential(
            nn.Linear(in_channels + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ))
        self.conv2 = PointNetConv(local_nn=nn.Sequential(
            nn.Linear(64 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ))
        self.gat1 = GATConv(128, 64, heads=4, concat=True)
        self.gat2 = GATConv(64 * 4, 128)
        self.fc = nn.Linear(128, out_channels)

    def forward(self, data):
        pos, x, edge_index = data.pos, data.x, data.edge_index
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)
        x = F.relu(self.gat1(x, edge_index))
        features = F.relu(self.gat2(x, edge_index))
        x = self.fc(features)
        return F.log_softmax(x, dim=-1), features

def compute_dice_score(pred, target, num_classes=5, eps=1e-6):
    dice_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(float)
        target_c = (target == c).astype(float)
        intersection = (pred_c * target_c).sum()
        dice = (2. * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_scores.append(dice)
    return np.mean(dice_scores)

def compute_iou(pred, target, num_classes=5, eps=1e-6):
    iou_scores = []
    for c in range(num_classes):
        pred_c = (pred == c).astype(float)
        target_c = (target == c).astype(float)
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        iou_scores.append(iou)
    return np.mean(iou_scores)

def visualize_segmentation(obj_file, pred_file, output_mesh_file):
    if not os.path.exists(obj_file) or not os.path.exists(pred_file):
        print(f"Missing file: {obj_file} or {pred_file}")
        return
    
    mesh = trimesh.load(obj_file)
    vertices = mesh.vertices
    faces = mesh.faces
    pred_labels = np.load(pred_file)

    if len(pred_labels) != len(vertices):
        print(f"Inconsistent number of vertices and predictions: {len(vertices)} vs {len(pred_labels)}")
        return

    colors = [
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255]
    ]
    vertex_colors = np.array([colors[label] for label in pred_labels])
    colored_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    os.makedirs(os.path.dirname(output_mesh_file), exist_ok=True)
    colored_mesh.export(output_mesh_file)
    print(f"Segmented mesh saved to: {output_mesh_file}")

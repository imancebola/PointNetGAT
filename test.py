import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score

from model import (
    PointNetGAT, load_data_from_folder, intra_class_variance_loss,
    compute_dice_score, compute_iou, visualize_segmentation
)

def test(model, test_loader, device, output_dir, obj_files, names, loss_type="combined"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    test_preds, test_targets = [], []
    test_loss = 0

    pred_dir = os.path.join(output_dir, "test_predictions")
    mesh_dir = os.path.join(output_dir, "test_segmented_meshes")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.to(device)
            output, features = model(data)

            ce_loss = criterion(output, data.y)
            cluster_loss = intra_class_variance_loss(features, data.y)
            if loss_type == "ce":
                loss = ce_loss
            elif loss_type == "combined": 
                loss = ce_loss + 0.1 * cluster_loss
            test_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            targets = data.y.cpu().numpy()

            test_preds.append(preds)
            test_targets.append(targets)

            name = names[i]
            np.save(os.path.join(pred_dir, f"{name}.npy"), preds)

            if name in obj_files:
                obj_file = obj_files[name]
                visualize_segmentation(obj_file, os.path.join(pred_dir, f"{name}.npy"), os.path.join(mesh_dir, f"{name}_segmented.obj"))

    preds_all = np.concatenate(test_preds)
    targets_all = np.concatenate(test_targets)

    acc = accuracy_score(targets_all, preds_all)
    dice = compute_dice_score(preds_all, targets_all)
    iou = compute_iou(preds_all, targets_all)
    avg_loss = test_loss / len(test_loader)

    metrics = [{
        "test_loss": avg_loss,
        "test_acc": acc,
        "test_dice": dice,
        "test_iou": iou
    }]

    pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, "test_metrics.csv"), index=False)
    print(f"Test complete. Metrics saved to {os.path.join(metrics_dir, 'test_metrics.csv')}.")

if __name__ == "__main__":
    loss_type = "ce"  #  "ce"- Cross-Entropy loss // "combined" intravariance and Cross-Entropy Loss
    data_dir = ""
    mesh_dir = ""
    processed_dir = os.path.join(data_dir, "preprocess_data", "test")
    obj_dir = os.path.join(mesh_dir, "")
    output_dir = os.path.join(data_dir, "output")
    model_path = os.path.join(output_dir, "models", "best_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data, test_names = load_data_from_folder(processed_dir)
    test_loader = DataLoader(test_data, batch_size=1)

    obj_files = {name: os.path.join(obj_dir, f"{name}.obj") for name in test_names if os.path.exists(os.path.join(obj_dir, f"{name}.obj"))}

    model = PointNetGAT(in_channels=3, out_channels=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    test(model, test_loader, device, output_dir, obj_files, test_names, loss_type)

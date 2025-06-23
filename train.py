import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader

from model import (
    PointNetGAT, load_data_from_folder, intra_class_variance_loss,
    compute_dice_score, compute_iou
)

def train(model, train_loader, val_loader, device, num_epochs, output_dir, val_names, loss_type="combined"):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    metrics = []

    model_dir = os.path.join(output_dir, "models")
    pred_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        train_preds, train_targets = [], []
        train_loss = 0

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output, features = model(data)

            ce_loss = criterion(output, data.y)
            cluster_loss = intra_class_variance_loss(features, data.y)
            if loss_type == "ce":
                loss = ce_loss
            elif loss_type == "combined":
                loss = ce_loss + 0.1 * cluster_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = output.argmax(dim=1).cpu().numpy()
            targets = data.y.cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets)

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_dice = compute_dice_score(np.array(train_preds), np.array(train_targets))
        train_iou = compute_iou(np.array(train_preds), np.array(train_targets))

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                data = data.to(device)
                output, features = model(data)

                ce_loss = criterion(output, data.y)
                cluster_loss = intra_class_variance_loss(features, data.y)
                if loss_type == "ce":
                    loss = ce_loss
                elif loss_type == "combined": 
                    loss = ce_loss + 0.1 * cluster_loss

                val_loss += loss.item()

                preds = output.argmax(dim=1).cpu().numpy()
                targets = data.y.cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets)

                np.save(os.path.join(pred_dir, f"{val_names[i]}_epoch{epoch}.npy"), preds)

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_dice = compute_dice_score(np.array(val_preds), np.array(val_targets))
        val_iou = compute_iou(np.array(val_preds), np.array(val_targets))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print(f"Best model saved: {os.path.join(model_dir, 'best_model.pth')}")

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_dice': train_dice,
            'train_iou': train_iou,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_dice': val_dice,
            'val_iou': val_iou,
        })

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

    pd.DataFrame(metrics).to_csv(os.path.join(metrics_dir, "metrics.csv"), index=False)
    print(f"Training complete. Metrics saved to {os.path.join(metrics_dir, 'metrics.csv')}.")

if __name__ == "__main__":
    # Set parameters
    loss_type = "ce"  # "ce"- Cross-Entropy loss // "combined" intravariance and Cross-Entropy Loss
    data_dir = ""
    processed_dir = "" #file with the .pt files
    processed_dir = os.path.join(data_dir, processed_dir)
    output_dir = os.path.join(data_dir, "output")
    num_epochs = 50
    batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, _ = load_data_from_folder(os.path.join(processed_dir, "train"))
    val_data, val_names = load_data_from_folder(os.path.join(processed_dir, "val"))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)


    model = PointNetGAT(in_channels=3, out_channels=5).to(device)

    train(model, train_loader, val_loader, device, num_epochs, output_dir, val_names, loss_type)

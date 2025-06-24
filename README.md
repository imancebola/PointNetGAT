# PointNetGAT

PointNetGAT is a deep learning architecture that combines **PointNet++** and **Graph Attention Networks (GAT)** to perform segmentation on 3D left atrial meshes. It is designed to process and label anatomical structures represented as triangular meshes.

![PointNetGAT](assets/pointarch.png)

## File Overview
| File/Folder            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `preprocessing.py`               | Prepares .pt files for PointNetGAT model training.              |
| `model.py`            | Configuration file of PointNetGAT model. Auxiliary and metrics (Dice, IoU) functions.      |
| `train..py`          | Runs train and validacion model.    |
| `test.py`            | Runs test model.      |
| `plot_meshes.py`       | Dash app to analyce the segmentation results.     |
| `rquierments.txt`            | Python dependencies for preprocessing and model training.                |

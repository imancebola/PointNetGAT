# PointNetGAT

PointNetGAT is a deep learning architecture that combines **PointNet++** and **Graph Attention Networks (GAT)** to perform segmentation on 3D left atrial meshes. It is designed to process and label anatomical structures represented as triangular meshes.

![PointNetGAT](assets/pointarch.png)

## File Overview
| File/Folder            | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `preprocessing.py`               | Prepares mesh files for training the PointNetGAT model              |
| `model.py`            | Configuration file of PointNetGAT model. Auxiliary and metrics (Dice, IoU) functions.      |
| `train.py`          | Trains the model and runs validation.    |
| `test.py`            | Evaluates the model on the test set.      |
| `plot_meshes.py`       | Dash app to analyce the segmentation results.     |
| `rquierments.txt`            | Python dependencies for preprocessing and model training.                |

## Installation (with Anaconda)

To install and run PointNetGAT locally, follow these steps:

1. Clone the repostory
```
git clone https://github.com/imancebola/PointNetGAT.git
```
2. Move to the project directory
```
cd PointNetGAT
```
3. Create a new environment (recommended)
```
conda create -n pointnetgat_env python=3.8
conda activate pointnetgat_env
```
4. Install dependencies from *requirements.txt*
```
pip install -r requirements.txt
```

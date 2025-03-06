import numpy as np
import torch
import matplotlib.pyplot as plt
from joint_mds3 import joint_mds3  # Importing JointMDS3 from joint_mds3.py
from utils.utils import geodesic_dist  # Import helper function for distance computation

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

import numpy as np

# ================================
# Load Synthetic Data (User must specify paths)
# ================================

# Enter the correct file paths for each dataset before running the script

## Synthetic dataset 1
s1_X1 = np.loadtxt("PATH_TO_FILE/s1_X1.txt")  # Update with actual file path
s1_X2 = np.loadtxt("PATH_TO_FILE/s1_X2.txt")  # Update with actual file path
s1_X3 = np.loadtxt("PATH_TO_FILE/s1_X3.txt")  # Update with actual file path
s1_y = np.loadtxt("PATH_TO_FILE/s1_y1.txt")  # Update with actual file path

## Synthetic dataset 2
s2_X1 = np.loadtxt("PATH_TO_FILE/s2_X1.txt")  # Update with actual file path
s2_X2 = np.loadtxt("PATH_TO_FILE/s2_X2.txt")  # Update with actual file path
s2_X3 = np.loadtxt("PATH_TO_FILE/s2_X3.txt")  # Update with actual file path
s2_y = np.loadtxt("PATH_TO_FILE/s2_y1.txt")  # Update with actual file path

## Synthetic dataset 3
s3_X1 = np.loadtxt("PATH_TO_FILE/s3_X1.txt")  # Update with actual file path
s3_X2 = np.loadtxt("PATH_TO_FILE/s3_X2.txt")  # Update with actual file path
s3_X3 = np.loadtxt("PATH_TO_FILE/s3_X3.txt")  # Update with actual file path
s3_y = np.loadtxt("PATH_TO_FILE/s3_y1.txt")  # Update with actual file path

# ================================
# Define Hyperparameters (Update as needed)
# ================================

n_components = 2  # Number of MDS components
alpha = 0.5  # Regularization parameter
eps = 1e-5  # Convergence tolerance
max_iter = 1000  # Maximum number of iterations
dissimilarity = "euclidean"  # Distance metric

# Optional dataset name and k-values
k_dict = {"synthetic": 5}  # Define dataset-specific k values
dataset = "synthetic"
k = k_dict[dataset]

# ================================
# Run Joint MDS on Synthetic Data
# ================================

Z1, Z2, Z3 = joint_mds3(
    core=s1_X1,  # Update with the correct core dataset
    B=s1_X2,  # Update with the correct B dataset
    C=s1_X3,  # Update with the correct C dataset
    n_components=n_components,
    alpha=alpha,
    eps=eps,
    max_iter=max_iter,
    dissimilarity=dissimilarity,
    labels=s1_y,  # Update with correct labels if available
    dataset=dataset,
    k_dict=k_dict,
    k=k,
    plot=True,  # Set to True to visualize embeddings
)

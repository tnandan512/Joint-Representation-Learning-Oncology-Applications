# Import necessary libraries
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom modules
from joint_mds import JointMDS
from utils.utils import plot_embedding, geodesic_dist
from sklearn.preprocessing import normalize
import utils.scores as scores

# Set random seed for reproducibility
torch.manual_seed(20)

# -------------------------------
# Load and Preprocess Data
# -------------------------------

# Load BraTS features (T2-FLAIR features extracted using PyRadiomics)
Y1 = pd.read_csv(
    "<path_to_braTS_data>"  # User should replace with the path to BraTS T2-FLAIR feature data
)
Y1.set_index(Y1.columns[0], inplace=True)  # Set the first column as the index
t2f_data = np.asarray(
    Y1.iloc[:, 1:-1]
)  # Extract relevant data excluding the last column

# Load RNA-Seq data for GBM and LGG samples
gbm_rnaseq = pd.read_csv(
    "<path_to_gbm_rnaseq_data>"
)  # User should replace with the path to GBM RNA-Seq data
gbm_rnaseq.set_index(
    gbm_rnaseq.columns[0], inplace=True
)  # Set the first column as the index

lgg_rnaseq = pd.read_csv(
    "<path_to_lgg_rnaseq_data>"
)  # User should replace with the path to LGG RNA-Seq data
lgg_rnaseq.set_index(
    lgg_rnaseq.columns[0], inplace=True
)  # Set the first column as the index

# Combine RNA-Seq datasets (GBM and LGG)
rnaseq_df = pd.concat(
    [lgg_rnaseq, gbm_rnaseq], axis=0
)  # Concatenate both RNA-Seq datasets
rna_data = np.asarray(
    rnaseq_df.iloc[:, 1:-1]
)  # Extract the RNA-Seq data excluding the last column

# -------------------------------
# Compute Dissimilarities Using Geodesic Distance
# -------------------------------

# Compute geodesic distances for T2-FLAIR features
D1 = geodesic_dist(
    t2f_data,
    k=50,
    metric="correlation",  # Calculate geodesic distance for T2-FLAIR data using correlation metric
)
# Compute geodesic distances for RNA-Seq data
D2 = geodesic_dist(
    rna_data,
    k=50,
    metric="correlation",  # Calculate geodesic distance for RNA-Seq data using correlation metric
)
D1 = torch.from_numpy(
    D1
).float()  # Convert the T2-FLAIR distance matrix to a PyTorch tensor
D2 = torch.from_numpy(
    D2
).float()  # Convert the RNA-Seq distance matrix to a PyTorch tensor

# -------------------------------
# Apply Joint MDS for Alignment
# -------------------------------

# Initialize JointMDS with specific parameters
JMDS = JointMDS(
    n_components=2,  # Reduce data to 2 dimensions (embedding)
    alpha=0.5,  # Regularization parameter for joint optimization
    eps=0.01,  # Convergence threshold
    max_iter=500,  # Maximum number of iterations
    eps_annealing=False,  # Disable epsilon annealing
    dissimilarity="precomputed",  # Use precomputed dissimilarity matrices
)

# Perform Joint MDS to compute the embeddings
Z1, Z2, P = JMDS.fit_transform(D1, D2)

# Convert PyTorch tensors to NumPy arrays for easier manipulation
Z1, Z2 = Z1.numpy(), Z2.numpy()

# -------------------------------
# Evaluate Alignment Performance
# -------------------------------

# Calculate the average FOSCTTM score for the alignment
fracs = scores.calc_domainAveraged_FOSCTTM(
    Z1, Z2
)  # Calculate domain averaged FOSCTTM score
print(
    "Average FOSCTTM score for this alignment with X1 onto X2 is:", np.mean(fracs)
)  # Print the FOSCTTM score

# Convert embeddings to DataFrames for saving
Z1 = pd.DataFrame(
    Z1, index=Y1.index, columns=["Dimension_1", "Dimension_2"]
)  # Create DataFrame for Z1
Z2 = pd.DataFrame(
    Z2, index=rnaseq_df.index, columns=["Dimension_1", "Dimension_2"]
)  # Create DataFrame for Z2

# Save the embeddings to text files
np.savetxt(
    "<path_to_save_Z1>", Z1
)  # User should replace with the path where to save Z1
np.savetxt(
    "<path_to_save_Z2>", Z2
)  # User should replace with the path where to save Z2

# -------------------------------
# Visualization
# -------------------------------

# Create labels for the embeddings: 0 for T2-FLAIR, 1 for RNA-Seq
labels_X1 = np.zeros(Z1.shape[0])  # Label for T2-FLAIR features
labels_X2 = np.ones(Z2.shape[0])  # Label for RNA-Seq features

# Combine embeddings and labels for plotting
Z_combined = np.vstack([Z1, Z2])  # Combine both Z1 and Z2
labels_combined = np.concatenate(
    [labels_X1, labels_X2]
)  # Combine the labels for Z1 and Z2

# Plot the embeddings
plt.figure(figsize=(8, 6))  # Set the figure size
plot_embedding(
    Z_combined, labels_combined, ax=plt.gca(), title="Joint MDS Embedding"
)  # Plot the embeddings
plt.xlabel("Joint MDS Component 1")  # X-axis label
plt.ylabel("Joint MDS Component 2")  # Y-axis label
plt.legend()  # Show legend
plt.savefig(
    "<path_to_save_plot>"
)  # User should replace with the path to save the plot image

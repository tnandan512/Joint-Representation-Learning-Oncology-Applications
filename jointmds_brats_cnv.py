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
t2f = pd.read_csv(
    "<path_to_braTS_data>"  # User should replace with the path to BraTS T2-FLAIR feature data
)
t2f.set_index(t2f.columns[-1], inplace=True)
t2f = t2f.rename(columns={"Cohort_Name": "Data"})
t2f["Data"] = t2f["Data"].replace({"TCGA-GBM": "GBM", "TCGA-LGG": "LGG"})
t2f_labels = t2f["Data"].tolist()
t2f_ids = t2f.index.tolist()
t2f.drop(columns=["Unnamed: 0", "Contrast_Type"], inplace=True)
threshold_ratio = 0.01
variances = t2f.var()
means = t2f.mean()
variance_to_mean_ratio = variances / means
selected_features = variance_to_mean_ratio[
    variance_to_mean_ratio >= threshold_ratio
].index
t2f = t2f[selected_features]
t2f_data = np.asarray(t2f.iloc[:, :-1])

# Load CNV data for GBM and LGG samples
cnv_df = pd.read_csv("<path_to_CNV_data>")
cnv_df.set_index(cnv_df.columns[0], inplace=True)
cnv_labels = cnv_df["Data"].tolist()
cnv_ids = cnv_df.index.tolist()
cnv_data = np.asarray(cnv_df.iloc[:, :-1])

# -------------------------------
# Compute Dissimilarities Using Geodesic Distance
# -------------------------------

# Compute geodesic distances for T2-FLAIR features
D1 = geodesic_dist(t2f_data, k=20, metric="correlation", mode="distance")
# Compute geodesic distances for CNV data
D2 = geodesic_dist(cnv_data, k=60, metric="correlation", mode="distance")
D1 = torch.from_numpy(
    D1
).float()  # Convert the T2-FLAIR distance matrix to a PyTorch tensor
D2 = torch.from_numpy(D2).float()  # Convert the CNV distance matrix to a PyTorch tensor

# -------------------------------
# Apply Joint MDS for Alignment
# -------------------------------

# Initialize JointMDS with specific parameters
JMDS = JointMDS(
    n_components=2,  # Reduce data to 2 dimensions (embedding)
    alpha=0.5,  # Regularization parameter for joint optimization
    eps=0.1,  # Convergence threshold
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
    Z1, index=t2f.index, columns=["Dimension_1", "Dimension_2"]
)  # Create DataFrame for Z1
Z2 = pd.DataFrame(
    Z2, index=cnv_df.index, columns=["Dimension_1", "Dimension_2"]
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

# Create labels for the embeddings: 0 for T2-FLAIR, 1 for CNV
labels_X1 = np.zeros(Z1.shape[0])  # Label for T2-FLAIR features
labels_X2 = np.ones(Z2.shape[0])  # Label for CNV features

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

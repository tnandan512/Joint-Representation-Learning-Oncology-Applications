# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joint_mds import JointMDS
from utils.utils import geodesic_dist
from sklearn.preprocessing import normalize
import utils.scores as scores
import itertools

# Set random seed for reproducibility
torch.manual_seed(100)

# Load RNA-Seq Data
# User should enter the correct path to the RNA-Seq data file
rna = pd.read_csv(
    "<path_to_rna_data>",  # Enter your RNA data file path here
    index_col=0,
)
rna_labels = rna["Data"].tolist()  # Extract labels for RNA-Seq data
rna_data = np.asarray(rna.iloc[:, :-1])  # Extract the data without the label column

# Log-transform and standardize RNA data
rna_data_log = np.log1p(rna_data)  # Apply log transformation to RNA data
rna_log_std = StandardScaler().fit_transform(
    rna_data_log
)  # Standardize the log-transformed RNA data

# Load CNV Data
# User should enter the correct path to the CNV data file
cnv = pd.read_csv(
    "<path_to_cnv_data>",  # Enter your CNV data file path here
    index_col=0,
)
cnv_labels = cnv["Data"].tolist()  # Extract labels for CNV data
cnv_data = np.asarray(cnv.iloc[:, :-1])  # Extract the data without the label column

# Parameter Grid for Grid Search
alpha_values = [0.1, 0.3, 0.5]  # Different values of alpha to test
eps_values = [0.01, 0.1, 0.5]  # Different values of epsilon to test
k_values = [20, 30, 50, 60]  # Different values of k (neighborhood size) to test
modes = ["connectivity", "distance"]  # Modes for graph construction
metrics = ["correlation", "minkowski"]  # Distance metrics for graph construction

# Storage for results
results = []  # List to store the results of grid search
best_params, best_foscttm = None, float(
    "inf"
)  # Initialize variables to track the best parameters
best_Z1, best_Z2 = None, None  # Variables to store the best embeddings

print("Starting grid search...")

# Perform grid search over all combinations of parameters
for alpha, eps, k1, k2, mode1, metric1, mode2, metric2 in itertools.product(
    alpha_values, eps_values, k_values, k_values, modes, metrics, modes, metrics
):
    torch.manual_seed(100)  # Ensuring consistency with a fixed seed

    # Normalize CNV and RNA data
    X1 = normalize(cnv_data, axis=1)  # Normalize CNV data along each row
    X2 = normalize(rna_log_std, axis=1)  # Normalize RNA data along each row

    # Compute geodesic distances between samples for CNV and RNA data
    D1 = torch.tensor(
        geodesic_dist(cnv_data, k=k1, metric=metric1, mode=mode1), dtype=torch.float32
    )  # Geodesic distances for CNV data
    D2 = torch.tensor(
        geodesic_dist(rna_log_std, k=k2, metric=metric2, mode=mode2),
        dtype=torch.float32,
    )  # Geodesic distances for RNA data

    # Perform JointMDS transformation to align the data in a lower-dimensional space
    JMDS = JointMDS(
        n_components=2,  # Reduce to 2 dimensions
        alpha=alpha,  # Regularization parameter
        eps=eps,  # Convergence threshold
        max_iter=500,  # Maximum number of iterations
        eps_annealing=False,  # No annealing of epsilon
        dissimilarity="precomputed",  # Use precomputed dissimilarity matrix
    )

    with torch.no_grad():  # Disable gradient tracking as no backpropagation is needed
        Z1, Z2, _, _ = JMDS.fit_transform(D1, D2)  # Fit the JointMDS model to the data

    Z1, Z2 = Z1.numpy(), Z2.numpy()  # Convert the embeddings to numpy arrays

    # Compute performance scores
    average_foscttm = np.mean(
        scores.calc_domainAveraged_FOSCTTM(Z1, Z2)
    )  # Calculate the FOSCTTM score
    acc = scores.transfer_accuracy(
        Z1, Z2, cnv_labels, rna_labels, 5
    )  # Calculate transfer accuracy between the datasets

    # Store the results of this iteration
    results.append(
        [alpha, eps, k1, k2, mode1, metric1, mode2, metric2, average_foscttm, acc]
    )

    # Track the best performing parameters based on FOSCTTM score
    if average_foscttm < best_foscttm:
        best_params = {
            "alpha": alpha,
            "eps": eps,
            "k1": k1,
            "k2": k2,
            "mode1": mode1,
            "metric1": metric1,
            "mode2": mode2,
            "metric2": metric2,
        }
        best_foscttm, best_Z1, best_Z2 = average_foscttm, Z1, Z2

# Convert the results list to a DataFrame for easy analysis and save to CSV
results_df = pd.DataFrame(
    results,
    columns=[
        "alpha",
        "eps",
        "k1",
        "k2",
        "mode1",
        "metric1",
        "mode2",
        "metric2",
        "FOSCTTM",
        "Acc",
    ],
)
results_df.to_csv(
    "gridsearch_cnv_rnaseq.csv", index=False
)  # Save results of grid search

# Save the best embeddings (Z1 and Z2) to CSV files
pd.DataFrame(best_Z1, columns=["Z1_1", "Z1_2"]).to_csv(
    "cnv_rna_best_Z1.csv", index=False
)  # Save the best embedding for CNV data
pd.DataFrame(best_Z2, columns=["Z2_1", "Z2_2"]).to_csv(
    "cnv_rna_best_Z2.csv", index=False
)  # Save the best embedding for RNA data

# Print the best parameters found during the grid search
print("Best Parameters:", best_params)

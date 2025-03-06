# Import necessary libraries
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import pandas as pd
import itertools
from joint_mds import JointMDS
from utils.utils import geodesic_dist
from sklearn.preprocessing import normalize
import utils.scores as scores

# Set the random seed for reproducibility
torch.manual_seed(100)

# Read in the RNA-Seq data
rna = pd.read_csv(
    "<path_to_rna_data>"
)  # User should replace with the path to the RNA-Seq data
rna.set_index(rna.columns[0], inplace=True)  # Set the first column as the index
rna_labels = rna["Data"].tolist()  # Extract labels from the 'Data' column
rna_data = np.asarray(
    rna.iloc[:, :-1]
)  # Extract the RNA data excluding the last column

# Log-transform the RNA data and standardize it
rna_data_log = np.log1p(rna_data)  # Log transformation to handle skewed data
scaler = StandardScaler()  # StandardScaler to standardize the data
rna_log_std = scaler.fit_transform(rna_data_log)  # Standardize the log-transformed data

# Read in the DNA Methylation data
methyl = pd.read_csv(
    "<path_to_methyl_data>"
)  # User should replace with the path to the DNA methylation data
methyl.set_index(methyl.columns[0], inplace=True)  # Set the first column as the index
methyl_labels = methyl["Data"].tolist()  # Extract labels from the 'Data' column
methyl_data = np.asarray(
    methyl.iloc[:, :-1]
)  # Extract the methylation data excluding the last column

# Define the parameter grid for the grid search
alpha_values = [0.1, 0.3, 0.5]  # Alpha values to test
eps_values = [0.01, 0.1, 0.5]  # Epsilon values to test
k1_values = [50, 60, 100]  # Nearest neighbors for the methylation data
k2_values = [50, 60, 100]  # Nearest neighbors for the RNA data
modes1 = ["connectivity", "distance"]  # Modes for methylation data
modes2 = ["connectivity", "distance"]  # Modes for RNA data
metrics1 = ["correlation", "minkowski"]  # Metrics for methylation data
metrics2 = ["correlation", "minkowski"]  # Metrics for RNA data

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(
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
    ]
)

# Set random seed for consistency in results
torch.random.manual_seed(100)
print("Starting grid search...")

# Initialize variables to track the best parameters and results
best_params = None
best_foscttm = float("inf")  # Initialize with a very large value to find the minimum
best_Z1, best_Z2 = None, None

# Perform grid search over all combinations of parameters
for alpha, eps, k1, k2, mode1, metric1, mode2, metric2 in itertools.product(
    alpha_values, eps_values, k1_values, k2_values, modes1, metrics1, modes2, metrics2
):
    # Ensure reproducibility by setting the random seed
    torch.manual_seed(100)

    # Normalize the methylation and RNA-Seq data
    X1 = normalize(methyl_data, axis=1)  # Normalize methylation data
    X2 = normalize(rna_log_std, axis=1)  # Normalize RNA data

    # Compute geodesic distances for methylation and RNA data using specified parameters
    D1 = geodesic_dist(
        methyl_data, k=k1, metric=metric1, mode=mode1
    )  # Geodesic distance for methylation
    D2 = geodesic_dist(
        rna_log_std, k=k2, metric=metric2, mode=mode2
    )  # Geodesic distance for RNA
    D1 = torch.from_numpy(D1).float()  # Convert the distance matrix to a PyTorch tensor
    D2 = torch.from_numpy(D2).float()  # Convert the distance matrix to a PyTorch tensor

    # Initialize and fit the JointMDS model
    JMDS = JointMDS(
        n_components=2,  # We want to reduce to 2 components
        alpha=alpha,  # Regularization parameter
        eps=eps,  # Convergence threshold
        max_iter=500,  # Maximum number of iterations
        eps_annealing=False,  # Whether to anneal epsilon
        dissimilarity="precomputed",  # Use precomputed distance matrix
    )

    # Perform Joint MDS optimization
    Z1, Z2, P, O = JMDS.fit_transform(D1, D2)  # Transform the data using Joint MDS
    Z1, Z2 = Z1.numpy(), Z2.numpy()  # Convert the results back to numpy arrays

    # Calculate the FOSCTTM score
    fracs = scores.calc_domainAveraged_FOSCTTM(
        Z1, Z2
    )  # Compute FOSCTTM score for the embeddings
    average_foscttm = np.mean(fracs)  # Take the average of the FOSCTTM score

    # Calculate transfer accuracy between the datasets
    acc = scores.transfer_accuracy(
        Z1, Z2, methyl_labels, rna_labels, 5
    )  # Calculate transfer accuracy with 5-fold cross-validation
    print("FOSCTTM:", average_foscttm, " Acc:", acc)

    # Store the results in the DataFrame
    results_df = results_df.append(
        {
            "alpha": alpha,
            "eps": eps,
            "k1": k1,
            "k2": k2,
            "mode1": mode1,
            "metric1": metric1,
            "mode2": mode2,
            "metric2": metric2,
            "FOSCTTM": average_foscttm,
            "Acc": acc,
        },
        ignore_index=True,
    )

    # Track the best parameters based on the lowest FOSCTTM score
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
        best_foscttm = average_foscttm  # Update the best FOSCTTM score
        best_Z1, best_Z2 = Z1, Z2  # Update the best embeddings

# Print the best parameters and results
print("DNA Methylation & RNAseq: Best Parameters")
print(best_params)

# Save the best embeddings (Z1 and Z2) to CSV files
pd.DataFrame(best_Z1, columns=["Z1_1", "Z1_2"]).to_csv(
    "methyl_rna_best_Z1.csv", index=False
)  # Save Z1 to CSV
pd.DataFrame(best_Z2, columns=["Z2_1", "Z2_2"]).to_csv(
    "methyl_rna_best_Z2.csv", index=False
)  # Save Z2 to CSV

# Save the grid search results to a CSV file
results_df.to_csv("gridsearch_methyl_rnaseq.csv", index=False)  # Save results to CSV

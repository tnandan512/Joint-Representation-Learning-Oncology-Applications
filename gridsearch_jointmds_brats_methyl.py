import torch
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

from joint_mds import JointMDS
from utils.utils import geodesic_dist
import utils.scores as scores

# Set random seed for reproducibility
torch.manual_seed(100)

# Load BraTS (Tumor Imaging) Data
# User: Replace the file path with the path to your 't2f_methylmatched_df.csv' file
t2f = pd.read_csv("/path/to/your/t2f_methylmatched_df.csv")
t2f.set_index(t2f.columns[-1], inplace=True)  # Set the last column as index
t2f = t2f.rename(columns={"Cohort_Name": "Data"})  # Rename 'Cohort_Name' to 'Data'
t2f["Data"] = t2f["Data"].replace(
    {"TCGA-GBM": "GBM", "TCGA-LGG": "LGG"}
)  # Update cohort labels
t2f_labels = t2f["Data"].tolist()  # Extract labels
t2f_ids = t2f.index.tolist()  # Extract sample ids
t2f.drop(
    columns=["Unnamed: 0", "Contrast_Type"], inplace=True
)  # Drop unnecessary columns
t2f_data = np.asarray(t2f.iloc[:, :-1])  # Extract features (excluding label column)

# Load DNA Methylation Data
# User: Replace the file path with the path to your 'methyl_bratsmatched_df.csv' file
methyl = pd.read_csv("/path/to/your/methyl_bratsmatched_df.csv")
methyl.set_index(methyl.columns[0], inplace=True)  # Set the first column as index
methyl_labels = methyl["Data"].tolist()  # Extract labels
methyl_data = np.asarray(
    methyl.iloc[:, :-1]
)  # Extract features (excluding label column)

# Define hyperparameter search space
alpha_values = [0.1, 0.3, 0.5]  # List of alpha values to test
eps_values = [0.01, 0.1, 0.5]  # List of epsilon values to test
k1_values = [20, 30, 50, 60]  # List of k1 values (for BraTS data) to test
k2_values = [20, 30, 50, 60]  # List of k2 values (for DNA methylation data) to test
modes1 = ["connectivity", "distance"]  # Modes for geodesic distance (BraTS)
modes2 = ["connectivity", "distance"]  # Modes for geodesic distance (Methylation)
metrics1 = ["correlation", "minkowski"]  # Metrics for geodesic distance (BraTS)
metrics2 = ["correlation", "minkowski"]  # Metrics for geodesic distance (Methylation)

# Initialize results storage (DataFrame to store results)
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

print("Starting grid search...")

# Variables to track the best performing parameters and embeddings
best_params = None
best_foscttm = float("inf")  # Lower FOSCTTM is better, so initialize with a high value
best_Z1, best_Z2 = None, None

# Perform grid search over all parameter combinations
for alpha, eps, k1, k2, mode1, metric1, mode2, metric2 in itertools.product(
    alpha_values, eps_values, k1_values, k2_values, modes1, metrics1, modes2, metrics2
):
    torch.manual_seed(100)  # Ensure consistency in results

    # Normalize the data (each feature vector is normalized to have unit norm)
    X1 = normalize(t2f_data, axis=1)
    X2 = normalize(methyl_data, axis=1)

    # Compute geodesic distance matrices for both datasets
    D1 = geodesic_dist(X1, k=k1, metric=metric1, mode=mode1)
    D2 = geodesic_dist(X2, k=k2, metric=metric2, mode=mode2)

    # Convert distance matrices to PyTorch tensors for compatibility with the model
    D1, D2 = torch.tensor(D1, dtype=torch.float32), torch.tensor(
        D2, dtype=torch.float32
    )

    # Initialize and fit the JointMDS model
    JMDS = JointMDS(
        n_components=2,  # Number of dimensions for the embedding
        alpha=alpha,  # Regularization parameter for the optimization
        eps=eps,  # Tolerance for stopping criteria
        max_iter=500,  # Maximum number of iterations
        eps_annealing=False,  # Whether to use annealing in the optimization
        dissimilarity="precomputed",  # We are providing precomputed distance matrices
    )

    # Perform joint MDS transformation to get embeddings Z1 and Z2
    Z1, Z2, P, O = JMDS.fit_transform(D1, D2)
    Z1, Z2 = Z1.numpy(), Z2.numpy()  # Convert embeddings to numpy arrays

    # Calculate evaluation metrics (FOSCTTM and Accuracy)
    foscttm = np.mean(scores.calc_domainAveraged_FOSCTTM(Z1, Z2))
    acc = scores.transfer_accuracy(Z1, Z2, t2f_labels, methyl_labels, 5)

    # Print results for this parameter combination
    print(
        f"Params: alpha={alpha}, eps={eps}, k1={k1}, k2={k2}, mode1={mode1}, metric1={metric1}, mode2={mode2}, metric2={metric2}"
    )
    print(f"FOSCTTM: {foscttm:.4f}, Accuracy: {acc:.4f}")

    # Store results in the DataFrame
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
            "FOSCTTM": foscttm,
            "Acc": acc,
        },
        ignore_index=True,
    )

    # Track best parameters based on FOSCTTM score (lower is better)
    if foscttm < best_foscttm:
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
        best_foscttm = foscttm  # Update the best score
        best_Z1, best_Z2 = Z1, Z2  # Save the best embeddings

# Display the best parameters found
print("Optimal Parameters for BraTS & DNA Methylation Alignment:")
print(best_params)

# Save the best embeddings and grid search results to CSV files
# User: Replace the file paths with where you want to save the results
pd.DataFrame(best_Z1, columns=["Z1_1", "Z1_2"]).to_csv(
    "/path/to/save/methyl_brats_best_Z1.csv", index=False
)
pd.DataFrame(best_Z2, columns=["Z2_1", "Z2_2"]).to_csv(
    "/path/to/save/methyl_brats_best_Z2.csv", index=False
)
results_df.to_csv("/path/to/save/gridsearch_brats_methyl.csv", index=False)

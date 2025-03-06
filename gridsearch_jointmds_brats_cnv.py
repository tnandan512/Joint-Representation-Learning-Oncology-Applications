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

# Read in CNV data (User should specify the file path for CNV data)
cnv = pd.read_csv("/path/to/cnv_bratsmatched_df.csv")  # Enter your file path here
cnv.set_index(cnv.columns[0], inplace=True)
cnv_labels = cnv["Data"].tolist()
cnv_ids = cnv.index.tolist()
cnv_data = np.asarray(cnv.iloc[:, :-1])

# Read in BraTS data (User should specify the file path for BraTS data)
t2f = pd.read_csv("/path/to/t2f_cnvmatched_df.csv")  # Enter your file path here
t2f.set_index(t2f.columns[-1], inplace=True)
t2f = t2f.rename(columns={"Cohort_Name": "Data"})
t2f["Data"] = t2f["Data"].replace({"TCGA-GBM": "GBM", "TCGA-LGG": "LGG"})
t2f_labels = t2f["Data"].tolist()
t2f_ids = t2f.index.tolist()
t2f.drop(columns=["Unnamed: 0", "Contrast_Type"], inplace=True)
t2f_data = np.asarray(t2f.iloc[:, :-1])

# Define the hyperparameter grid for grid search
alpha_values = [0.1, 0.3, 0.5]
eps_values = [0.01, 0.1, 0.5]
k1_values = [20, 30, 50, 60]
k2_values = [20, 30, 50, 60]
modes1 = ["connectivity", "distance"]
modes2 = ["connectivity", "distance"]
metrics1 = ["correlation", "minkowski"]
metrics2 = ["correlation", "minkowski"]

# Initialize a DataFrame to store results from the grid search
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
torch.random.manual_seed(100)
print("Starting grid search loop...")

# Variables to store the best parameters and results
best_params = None
best_foscttm = float("inf")  # Initialize with a large value
best_Z1, best_Z2 = None, None

# Grid search over all combinations of hyperparameters
for alpha, eps, k1, k2, mode1, metric1, mode2, metric2 in itertools.product(
    alpha_values, eps_values, k1_values, k2_values, modes1, metrics1, modes2, metrics2
):
    # Ensure results are consistent
    torch.manual_seed(100)

    # Compute geodesic distance matrices for BraTS and CNV data
    D1 = geodesic_dist(t2f_data, k=k1, metric=metric1, mode=mode1)
    D2 = geodesic_dist(cnv_data, k=k2, metric=metric2, mode=mode2)
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    # Initialize and fit the JointMDS model
    JMDS = JointMDS(
        n_components=2,
        alpha=alpha,
        eps=eps,
        max_iter=500,
        eps_annealing=False,
        dissimilarity="precomputed",
    )

    Z1, Z2, P, O = JMDS.fit_transform(D1, D2)
    Z1, Z2 = Z1.numpy(), Z2.numpy()

    # Calculate evaluation metrics: FOSCTTM and accuracy
    fracs = scores.calc_domainAveraged_FOSCTTM(Z1, Z2)
    average_foscttm = np.mean(fracs)
    acc = scores.transfer_accuracy(Z1, Z2, t2f_labels, cnv_labels, 5)

    # Print progress
    print("FOSCTTM: ", average_foscttm, " Accuracy: ", acc)

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

    # Track the best parameters based on FOSCTTM
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
        best_acc = acc
        best_Z1, best_Z2 = Z1, Z2

# Display the best parameters found
print("BraTS & CNV: Best Parameters")
print(best_params)

# Save the embeddings corresponding to the best parameters
# (User should specify where to save the CSV files)
pd.DataFrame(best_Z1, columns=["Z1_1", "Z1_2"]).to_csv(
    "best_Z1_cnvbrats.csv", index=False
)  # Enter your file path here
pd.DataFrame(best_Z2, columns=["Z2_1", "Z2_2"]).to_csv(
    "best_Z2_cnvbrats.csv", index=False
)  # Enter your file path here

# Save the results from the grid search
results_df.to_csv("gridsearch_brats_cnv.csv", index=False)  # Enter your file path here

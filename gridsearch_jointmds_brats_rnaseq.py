import sys
import os
import torch
import numpy as np
import pandas as pd
from joint_mds import JointMDS
from utils.utils import plot_embedding, geodesic_dist
from sklearn.preprocessing import normalize
import utils.scores as scores
import itertools

torch.manual_seed(42)

# Read in data
## Brats features from pyradiomics
X1 = pd.read_csv(
    "path_to_t1c_matched_84ids.csv"
)  # Enter path to the t1c_matched_84ids.csv file
X1.set_index(X1.columns[0], inplace=True)
t1c_data = np.asarray(X1.iloc[:, 1:-1])

Y1 = pd.read_csv(
    "path_to_t2f_matched_84ids.csv"
)  # Enter path to the t2f_matched_84ids.csv file
Y1.set_index(Y1.columns[0], inplace=True)
t2f_data = np.asarray(Y1.iloc[:, 1:-1])

# RNASeq data
gbm_rnaseq = pd.read_csv(
    "path_to_rnaseq_gbm_matched_84ids.csv"
)  # Enter path to the rnaseq_gbm_matched_84ids.csv file
gbm_rnaseq.set_index(gbm_rnaseq.columns[0], inplace=True)

lgg_rnaseq = pd.read_csv(
    "path_to_rnaseq_lgg_matched_84ids.csv"
)  # Enter path to the rnaseq_lgg_matched_84ids.csv file
lgg_rnaseq.set_index(lgg_rnaseq.columns[0], inplace=True)

rnaseq_df = pd.concat([lgg_rnaseq, gbm_rnaseq], axis=0)
rna_data = np.asarray(rnaseq_df.iloc[:, 1:-1])

# Define the parameter grid
alpha_values = [0.1, 0.3, 0.3, 0.5, 1]
eps_values = [0.01, 0.05, 0.1, 0.5, 1]
k_values = [20, 30, 50, 25]

# Initialize a DataFrame to store results
results_df = pd.DataFrame(columns=["alpha", "eps", "k", "FOSCTTM"])

for alpha, eps, k in itertools.product(alpha_values, eps_values, k_values):
    # Perform the grid search with current parameters
    JMDS = JointMDS(
        n_components=2,
        alpha=0.5,
        eps=0.01,
        max_iter=500,
        eps_annealing=False,
        dissimilarity="precomputed",
    )

    # Update the geodesic distance with the current k value
    D1 = geodesic_dist(t2f_data, k=50, metric="correlation")
    D2 = geodesic_dist(rna_data, k=50, metric="correlation")
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

    Z1, Z2, P = JMDS.fit_transform(D1, D2)
    Z1, Z2 = Z1.numpy(), Z2.numpy()

    # Calculate FOSCTTM
    fracs = scores.calc_domainAveraged_FOSCTTM(Z1, Z2)
    average_foscttm = np.mean(fracs)

    # Store the results in the DataFrame
    results_df = results_df.append(
        {"alpha": alpha, "eps": eps, "k": k, "FOSCTTM": average_foscttm},
        ignore_index=True,
    )

# Display the results
results_df.to_csv("T2F_RNA_Seed42.txt", sep="\t", index=False)  # The output file path

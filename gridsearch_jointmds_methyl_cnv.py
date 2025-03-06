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

# from scipy.spatial.distance import pdist
import itertools

torch.manual_seed(100)

# Read in data

## CNV
df_cnv_path = "# UPDATE: Path to CNV data file"
cnv = pd.read_csv(df_cnv_path)
cnv.set_index(cnv.columns[0], inplace=True)
cnv_labels = cnv["Data"].tolist()
cnv_ids = cnv.index.tolist()
cnv_data = np.asarray(cnv.iloc[:, :-1])

## DNA Methylation data
df_methyl_path = "# UPDATE: Path to Methylation data file"
methyl = pd.read_csv(df_methyl_path)
methyl.set_index(methyl.columns[0], inplace=True)
methyl_labels = methyl["Data"].tolist()
methyl_data = np.asarray(methyl.iloc[:, :-1])

# Define the parameter grid
alpha_values = [0.1, 0.3, 0.5]
eps_values = [0.01, 0.1, 0.5]
k1_values = [20, 30, 50, 60]
k2_values = [20, 30, 50, 60]
modes1 = ["connectivity", "distance"]
modes2 = ["connectivity", "distance"]
metrics1 = ["correlation", "minkowski"]
metrics2 = ["correlation", "minkowski"]

# Initialize a DataFrame to store results
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
print("Starting bigger loop...")

best_params = None
best_foscttm = float("inf")  # Initialize with a large value
best_Z1, best_Z2 = None, None

for alpha, eps, k1, k2, mode1, metric1, mode2, metric2 in itertools.product(
    alpha_values, eps_values, k1_values, k2_values, modes1, metrics1, modes2, metrics2
):
    # Perform the grid search with current parameters
    torch.manual_seed(100)

    X1 = normalize(methyl_data, axis=1)
    X2 = normalize(cnv_data, axis=1)

    D1 = geodesic_dist(methyl_data, k=k1, metric=metric1, mode=mode1)
    D2 = geodesic_dist(cnv_data, k=k2, metric=metric2, mode=mode2)
    D1 = torch.from_numpy(D1).float()
    D2 = torch.from_numpy(D2).float()

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

    # Calculate FOSCTTM
    fracs = scores.calc_domainAveraged_FOSCTTM(Z1, Z2)
    average_foscttm = np.mean(fracs)

    acc = scores.transfer_accuracy(Z1, Z2, methyl_labels, cnv_labels, 5)
    print("FOSCTTM: ", average_foscttm, " Acc: ", acc)

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
        best_foscttm = average_foscttm
        best_Z1, best_Z2 = Z1, Z2

# Display the results
print("DNA Methylation & CNV: Best Parameters")
print(best_params)

# Save embeddings corresponding to the best parameters
pd.DataFrame(best_Z1, columns=["Z1_1", "Z1_2"]).to_csv(
    "methyl_cnv_best_Z1.csv", index=False
)
pd.DataFrame(best_Z2, columns=["Z2_1", "Z2_2"]).to_csv(
    "methyl_cnv_best_Z2.csv", index=False
)

results_df.to_csv("gridsearch_methyl_cnv.csv", index=False)

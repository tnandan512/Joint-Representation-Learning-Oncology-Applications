import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Update file paths before running
GBM_PATH = "/path/to/TCGA-GBM.htseq_fpkm.tsv"  # Change input file path
LGG_PATH = "/path/to/TCGA-LGG.htseq_fpkm.tsv"  # Change input file path

# Load GBM dataset
gbm_df = pd.read_csv(GBM_PATH, sep="\t")
gbm_df = gbm_df.T
gbm_df.columns = gbm_df.iloc[0]
gbm_df = gbm_df.iloc[1:]
gbm_df.reset_index(inplace=True)
gbm_df.set_index(gbm_df.columns[0], inplace=True)
gbm_df["Data"] = "GBM"

# Load LGG dataset
lgg_df = pd.read_csv(LGG_PATH, sep="\t")
lgg_df = lgg_df.T
lgg_df.columns = lgg_df.iloc[0]
lgg_df = lgg_df.iloc[1:]
lgg_df.reset_index(inplace=True)
lgg_df.set_index(lgg_df.columns[0], inplace=True)
lgg_df["Data"] = "LGG"

# Combine both datasets
rnaseq_df = pd.concat([lgg_df, gbm_df], axis=0)

# Extract numerical RNA data (excluding labels)
rna_data = rnaseq_df.iloc[:, :-1].values
rna_data = np.asarray(rna_data, dtype=np.float64)

# Feature-wise standardization
column_variances = rnaseq_df.var()
top_columns = column_variances.sort_values(ascending=False).head(5000).index
rna_5000 = rnaseq_df[top_columns]
rna_5000 = rna_5000.apply(pd.to_numeric, errors="coerce")
rna_data = rna_5000.iloc[:, :-1].values
rna_data = np.asarray(rna_data, dtype=np.float64)

# Log transformation
rna_data_log = np.log1p(rna_data)

# Standardization
scaler = StandardScaler()
rna_log_std = scaler.fit_transform(rna_data_log)

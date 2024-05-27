#######################
### Importing Libraries
#######################
import sys
sys.path.append("DynProfiler")
import dynprofiler as dynpro

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)


#######################
### Configurations
#######################
base_path = Path("fig2/data")
outdir = base_path.joinpath("output")
train_params = yaml.safe_load(open(base_path.joinpath("input", "train_params.yml")))
model_params = yaml.safe_load(open(base_path.joinpath("input", "model_params.yml")))
dataset_params = yaml.safe_load(open(base_path.joinpath("input", "dataset_params.yml")))

train_params["prefix"] = "0"

#######################
### Load Data
#######################
# We conducted self-supervised learning on pan cancer. Please refer to our previos paper (Imoto et al., 2022) for the generation of simulated dynamics.
pan_cancer_mean = np.load(base_path.joinpath("input", "0_all_simulations_mean.npy"))
pan_cancer_std = np.load(base_path.joinpath("input", "0_all_simulations_std.npy"))
ids = np.load(base_path.joinpath("input", "0_all_ids.npy"), allow_pickle=True)
cancer_types = np.load(base_path.joinpath("input", "0_cancer_types.npy"))
brca_clinical = pd.read_csv(base_path.joinpath("input", "0_BRCA_clinical.csv"))


#######################
### Embedding
#######################
dynpro.embed({"mean": pan_cancer_mean, "std": pan_cancer_std},
             outdir,
             **train_params, **model_params, **dataset_params)


#######################
### Visualization
#######################
embs = np.load(base_path.joinpath("output", "0_1DCNN_SimSiam_seed10_epochs200.npy"))
brca_ind = np.where(cancer_types == "BRCA")[0]
brca_embs = embs[brca_ind].copy()
brca_mean = pan_cancer_mean[brca_ind].copy()
brca_ids = ids[brca_ind].copy()

# UMAP
umap_coord = UMAP(random_state=0).fit_transform(brca_embs)
fig, (ax1, ax2) = plt.subplots(figsize=(11,5), ncols=2)
sns.scatterplot(x=umap_coord[:, 0], y=umap_coord[:, 1], color="#2b2b2b", ax=ax1)
sns.scatterplot(x=umap_coord[:, 0], y=umap_coord[:, 1], hue=brca_clinical["BRCA_Subtype_PAM50"].values, palette="husl", ax=ax2)
ax1.set_xlabel("UMAP1")
ax1.set_ylabel("UMAP2")
ax2.set_xlabel("UMAP1")
ax2.set_ylabel("UMAP2")
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.tight_layout()
fig.savefig(base_path.joinpath("output", "0_BRCA_embedding_umap.png"))
plt.close()
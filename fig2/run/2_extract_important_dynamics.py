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

train_params["prefix"] = "2"
train_params["lr"] = 1e-3
train_params["n_splits"] = 5
train_params["pretrained_path"] = str(outdir.joinpath("0_1DCNN_SimSiam_seed10_epochs200.pth"))
train_params["fix_base"] = True
train_params["max_patient"] = 10


#######################
### Load Data
#######################
pan_cancer_mean = np.load(base_path.joinpath("input", "0_all_simulations_mean.npy"))
ids = np.load(base_path.joinpath("input", "0_all_ids.npy"), allow_pickle=True)
cancer_types = np.load(base_path.joinpath("input", "0_cancer_types.npy"))
brca_ind = np.where(cancer_types == "BRCA")[0]
brca_mean = pan_cancer_mean[brca_ind].copy()

brca_clinical = pd.read_csv(base_path.joinpath("input", "0_BRCA_clinical.csv"))
# Exclude TCGA-C8-A12T because this had no follow-up and was dropped in survival analysis
is_drop = brca_clinical["submitter_id"].values == "TCGA-C8-A12T"
brca_mean = brca_mean[~is_drop]

labels = np.load(base_path.joinpath("output", "1_risk_labels.npy"))


#######################
### Extract important dynamics
#######################
dynpro.interpret({"mean": brca_mean}, outdir, labels, **train_params, **model_params, **dataset_params)

attrs_ave = np.load(base_path.joinpath("output", "2_1DCNN_seed10_attribution_class1.npy"))
fig, ax = plt.subplots(figsize=(6,5))
t = np.arange(brca_mean.shape[-1])
for i in range(len(attrs_ave)):
    if attrs_ave[i].mean() > 0:
        c = sns.color_palette('coolwarm')[-1]
    else:
        c = sns.color_palette('coolwarm')[0]
    if np.max(attrs_ave[i]) > 0.5:
        ax.plot(t, attrs_ave[i], c=c)
    else:
        ax.plot(t, attrs_ave[i], c='grey', alpha=0.2)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Attribution')
fig.savefig(base_path.joinpath("output", "2_1DCNN_seed10_attribution.png"))
plt.close()
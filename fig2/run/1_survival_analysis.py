#######################
### Importing Libraries
#######################
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

from sksurv.datasets import load_breast_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from umap import UMAP

warnings.simplefilter("ignore", UserWarning)
#######################
### Configurations
#######################
base_path = Path("fig2/data")
outdir = base_path.joinpath("output")

#######################
### Load Data
#######################
ids = np.load(base_path.joinpath("input", "0_all_ids.npy"), allow_pickle=True)
cancer_types = np.load(base_path.joinpath("input", "0_cancer_types.npy"))
brca_clinical = pd.read_csv(base_path.joinpath("input", "0_BRCA_clinical.csv"))

embs = np.load(base_path.joinpath("output", "0_1DCNN_SimSiam_seed10_epochs200.npy"))
brca_ind = np.where(cancer_types == "BRCA")[0]
brca_embs = embs[brca_ind].copy()


# get n-year surval information
def get_n_year_survival(row, n):
    if row['days_to_death'] <= n * 365:
        return 'Dead'
    elif row['days_to_last_follow_up'] <= n * 365:
        return 'Omit'
    else:
        return 'Alive'
for y in [3,5,7]:
    brca_clinical[f'{y}_year_survival'] = brca_clinical.apply(lambda row: get_n_year_survival(row, {y}), axis=1)

brca_clinical['OS'] = brca_clinical['days_to_death'].apply(lambda x: True if not np.isnan(x) else False)
brca_clinical['OS_time'] = brca_clinical.apply(lambda row: row['days_to_last_follow_up'] if np.isnan(row['days_to_death']) else row['days_to_death'], axis=1)
# Exclude samples whose OS is 0 (because they do not have follow-up information)
incl_ind = np.where(brca_clinical['OS_time'] != 0)[0]
brca_clinical = brca_clinical.iloc[incl_ind].reset_index(drop=True)
brca_embs = brca_embs[incl_ind]


#######################
### Survival Analysis
#######################
# Dummy data in order to get scikit-survival's specific dtype
_, y = load_breast_cancer()
dt = y[0].dtype
OS_arr = []
for e, d in zip(brca_clinical['OS'], brca_clinical['OS_time']):
    OS_arr.append((bool(e), float(d)))
OS_arr = np.array(OS_arr, dtype=dt)

brca_umap = UMAP(random_state=42, n_components=6).fit_transform(brca_embs)
for i in np.arange(0.1, 1.1, 0.1):
    coxnet_pipe = make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=i, alpha_min_ratio=0.01, max_iter=100)) # Lasso: l1_ratio=1
    warnings.simplefilter("ignore", UserWarning)
    coxnet_pipe.fit(brca_umap, OS_arr)
    estimated_alphas = coxnet_pipe.named_steps["coxnetsurvivalanalysis"].alphas_
    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    gcv = GridSearchCV(
        make_pipeline(StandardScaler(), CoxnetSurvivalAnalysis(l1_ratio=i)),
        param_grid={"coxnetsurvivalanalysis__alphas": [[v] for v in estimated_alphas]},
        cv=cv,
        error_score=0.5,
        n_jobs=1,
    ).fit(brca_umap, OS_arr)

    cv_results = pd.DataFrame(gcv.cv_results_)
    alphas = cv_results.param_coxnetsurvivalanalysis__alphas.map(lambda x: x[0])
    mean = cv_results.mean_test_score
    std = cv_results.std_test_score
    print(f'ratio: {i}, max: {mean.max()}')
best_model = gcv.best_estimator_.named_steps["coxnetsurvivalanalysis"]
risk_score = np.sum(brca_umap * best_model.coef_.reshape(1, -1), axis=1)

# visulaliize ROC curve
fig, ax = plt.subplots(figsize=(6,6))
for y in [3, 5, 7]:
    ind = np.where(brca_clinical[f'{y}_year_survival'] != 'Omit')[0]
    y_true = brca_clinical[f'{y}_year_survival'].apply(lambda x: 1 if x=='Dead' else 0)[ind]
    y_score = risk_score[ind]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    ax.plot(fpr, tpr, label=f'AUC at {y}-year: {auc_score:.4f}')
ax.plot([0,1], [0,1], linestyle='dashed',c='grey')
ax.set_xlabel('FPR: False positive rate')
ax.set_ylabel('TPR: True positive rate')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
fig.savefig(base_path.joinpath("output", "1_BRCA_survival_auroc.png"))
plt.close()
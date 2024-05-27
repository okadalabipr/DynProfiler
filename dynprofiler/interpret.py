import math
from pathlib import Path
from datetime import datetime
import warnings
from typing import Any, Iterable, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from captum.attr import (
    DeepLift,
    visualization as viz,
)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

from .models import SimSiam, CNN1d, LSTM, GRU, VanillaRNN
from .trains import cross_validation

def set_param(name_, default_, **kwargs):
    param = kwargs.get(name_, None)
    if param is None:
        param = default_
    return param

# https://github.com/pytorch/captum/blob/b1a9830285c659556a6969c79ec720840397595e/captum/attr/_utils/visualization.py#L71
from numpy import ndarray
def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]

def _normalize_scale(attr: ndarray, scale_factor: float):
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)

def interpret(data, outdir, labels, **kwargs):
    if labels.dtype == "float":
        raise ValueError(f"Currently only classification task is supported but labels have float values.") 

    outdir = Path(outdir)

    if isinstance(data, dict):
        inp_mean = data["mean"]
    else:
        inp_mean = data
    
    prefix = set_param("prefix", datetime.now().strftime("%Y%m%d"), **kwargs)
    
    seed = set_param("seed", 42, **kwargs)
    n_splits = set_param("n_splits", 3, **kwargs)
    n_classes = len(np.unique(labels))
    task = "binary" if n_classes == 2 else "multi"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Correct labels
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    if not np.all(labels == new_labels):
        pd.DataFrame({"label": le.classes_, "label_for_model": np.arange(len(le.classes_))}).to_csv(outdir.joinpath(f"{prefix}_corresp_table.csv"), index=False)
        print("Convert labels so that they starts with 0 and the correspondence table was saved by name 'corresp_table.csv'.")
    
    ##################
    # cross_validation
    ##################
    fold_arr = cross_validation(data, outdir, new_labels, **kwargs)
    

    ##################
    # DeepLift
    ##################
    # Define models
    model_name = set_param("name", "1DCNN", **kwargs)
    if model_name.lower() not in ["1dcnn", "lstm", "gru", "rnn"]:
        raise Exception(f"Error: {model_name} is not in the allowed list [1DCNN, LSTM, GRU, RNN].")
    seq_len = inp_mean.shape[-1]
    in_feat = inp_mean.shape[1]
    emb_size = set_param("emb_size", 2 ** (math.ceil(math.log2(in_feat))), **kwargs)

    if model_name.lower() == "1dcnn":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size", "seq_len"]}
        model = CNN1d(in_feat, emb_size, seq_len, **inp_kwargs)
        inp_mean = torch.from_numpy(inp_mean).float()
    elif model_name.lower() == "lstm":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        model = LSTM(in_feat, emb_size, **inp_kwargs)
        inp_mean = torch.from_numpy(inp_mean).float().permute([0, 2, 1])
    elif model_name.lower() == "gru":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        model = GRU(in_feat, emb_size, **inp_kwargs)
        inp_mean = torch.from_numpy(inp_mean).float().permute([0, 2, 1])
    elif model_name.lower() == "rnn":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        model = VanillaRNN(in_feat, emb_size, **inp_kwargs)
        inp_mean = torch.from_numpy(inp_mean).float().permute([0, 2, 1])

    if n_classes == 2:
        # binary classification
        model._modules["predictor"] = nn.Linear(emb_size, 1)
    else:
        # multiclass classification
        model._modules["predictor"] = nn.Linear(emb_size, n_classes)
    model.to(device)
    model.eval()

    
    if task == "binary":
        attrs_ave_results = {"1": np.zeros(shape=(in_feat, seq_len))}
    else:
        attrs_ave_results = {f"{i}": np.zeros(shape=(in_feat, seq_len)) for i in range(n_classes)}

    for fold in range(n_splits):
        val_ind = np.where(fold_arr == fold)[0]
        model.load_state_dict(torch.load(outdir.joinpath(f'{prefix}_{model_name}_seed{seed}_fold{fold}.pth'), map_location=device))
        
        dl = DeepLift(model, multiply_by_inputs=True)
        _x = inp_mean[val_ind].to(device)
        
        fold_labels = new_labels[val_ind]
        if task == "binary":
            isPositive = np.where(fold_labels == 1)[0]
            attrs = dl.attribute(_x, target=0).detach().to("cpu").numpy()
            attrs = attrs[isPositive].mean(axis=0)
            attrs_ave_results["1"] += attrs / n_splits
        else:
            for i in range(n_classes):
                ind = np.where(fold_labels == i)[0]
                attrs = dl.attribute(_x[ind], target=i).detach().to("cpu").numpy()
                attrs = attrs.mean(axis=0)
                attrs_ave_results[f"{i}"] += attrs / n_splits
    
    outlier_perc = 2

    for k, v in attrs_ave_results.items():
        threshold = _cumulative_sum_threshold(np.abs(v), 100 - outlier_perc)
        v = _normalize_scale(v, threshold)
        np.save(outdir.joinpath(f'{prefix}_{model_name}_seed{seed}_attribution_class{k}'), v)
    
    return None















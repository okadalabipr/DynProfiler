import math
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader

from .utils import seed_everything, AverageMeter, asMinutes
from .models import SimSiam, CNN1d, LSTM, GRU, VanillaRNN
from .datasets import CustomDataset, SupervisedDataset

def set_param(name_, default_, **kwargs):
    param = kwargs.get(name_, None)
    if param is None:
        param = default_
    return param

def train_fn(model, train_loader, optimizer, criterion, device):
    model.train()
    losses = AverageMeter()
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)

        optimizer.zero_grad()
        pred = model(x)
        if pred.size(-1) == 1:
            # binary task
            loss = criterion(pred.view(-1, 1), y.view(-1, 1).float())
        else:
            # multiclass task
            loss = criterion(pred, y)
        
        losses.update(loss.item(), batch_size)
        loss.backward()
        optimizer.step()
    return losses.avg

def valid_fn(model, valid_loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    for step, (x, y) in enumerate(valid_loader):
        x = x.to(device)
        y = y.to(device)
        batch_size = x.size(0)
        with torch.no_grad():
            pred = model(x)
        if pred.size(-1) == 1:
            # binary task
            loss = criterion(pred.view(-1, 1), y.view(-1, 1).float())
            losses.update(loss.item(), batch_size)
            preds.append(pred.sigmoid().detach().to('cpu').numpy())
        else:
            # multiclass task
            loss = criterion(pred, y)
            losses.update(loss.item(), batch_size)
            preds.append(pred.argmax(dim=-1).detach().to('cpu').numpy())
        
    preds = np.concatenate(preds)
    return losses.avg, preds

def train(data, outdir, labels, trn_ind, val_ind, fold=None, **kwargs):
    if labels.dtype == "float":
        raise ValueError(f"Currently only classification task is supported but labels have float values.") 

    outdir = Path(outdir)

    if isinstance(data, dict):
        inp_mean = data["mean"]
    else:
        inp_mean = data
    
    prefix = set_param("prefix", datetime.now().strftime("%Y%m%d"), **kwargs)

    # trainig setting
    seed = set_param("seed", 42, **kwargs)
    epochs = set_param("epochs", 200, **kwargs)
    lr = float(set_param("lr", 1e-4, **kwargs))
    batch_size = set_param("batch_size", 16, **kwargs)
    num_workers = set_param("num_workers", 2, **kwargs)
    n_splits = set_param("n_splits", 3, **kwargs)
    pretrained_path = set_param("pretrained_path", None, **kwargs)
    fix_base = set_param("fix_base", True, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_classes = len(np.unique(labels))
    task = "binary" if n_classes == 2 else "multi"
    max_patient = set_param("max_patient", 3, **kwargs)

    # Correct labels
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    if not np.all(labels == new_labels):
        pd.DataFrame({"label": le.classes_, "label_for_model": np.arange(len(le.classes_))}).to_csv(outdir.joinpath(f"{prefix}_corresp_table.csv"), index=False)
        print("Convert labels so that they starts with 0 and the correspondence table was saved by name 'corresp_table.csv'.")
    
    # Define dataset
    trn_ds = SupervisedDataset(inp_mean=inp_mean[trn_ind], labels=new_labels[trn_ind], train=True, **kwargs)
    trn_dl = DataLoader(trn_ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    val_ds = SupervisedDataset(inp_mean=inp_mean[val_ind], labels=new_labels[val_ind], train=True, **kwargs)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    
    # Define models
    model_name = set_param("name", "1DCNN", **kwargs)
    if model_name.lower() not in ["1dcnn", "lstm", "gru", "rnn"]:
        raise Exception(f"Error: {model_name} is not in the allowed list [1DCNN, LSTM, GRU, RNN].")
    seq_len = inp_mean.shape[-1]
    in_feat = inp_mean.shape[1]
    emb_size = set_param("emb_size", 2 ** (math.ceil(math.log2(in_feat))), **kwargs)

    if model_name.lower() == "1dcnn":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size", "seq_len"]}
        base_model = CNN1d(in_feat, emb_size, seq_len, **inp_kwargs)
    elif model_name.lower() == "lstm":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        base_model = LSTM(in_feat, emb_size, **inp_kwargs)
    elif model_name.lower() == "gru":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        base_model = GRU(in_feat, emb_size, **inp_kwargs)
    elif model_name.lower() == "rnn":
        inp_kwargs = {k: v for k, v in kwargs.items() if k not in ["in_feat", "emb_size"]}
        base_model = VanillaRNN(in_feat, emb_size, **inp_kwargs)
    
    seed_everything(seed)
    model = SimSiam(base_model)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path, map_location="cpu"))
    model = model.model
    if n_classes == 2:
        # binary classification
        model._modules["predictor"] = nn.Linear(emb_size, 1)
    else:
        # multiclass classification
        model._modules["predictor"] = nn.Linear(emb_size, n_classes)
    model.to(device)
    model.train()
    
    # Define Loss function
    if task == "binary":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Set optimizer
    if fix_base:
        params = []
        for name, param in model.named_parameters():
            if name.startswith("predictor"):
                param.requires_grad = True
                params.append(param)
            else:
                param.requires_grad = False
    else:
        params = model.parameters()
    optimizer = Adam(params, lr=lr)
    
    # Start Train
    seed_everything(seed)
    start = end = time.time()
    loss_log = []
    best_loss = 9999.
    n_patient = 0
    for epoch in range(1, epochs+1):
        # train
        trn_losses = train_fn(model, trn_dl, optimizer, criterion, device)
        # valid
        val_losses, preds = valid_fn(model, val_dl, criterion, device)
        if task == "binary":
            score = roc_auc_score(new_labels[val_ind], preds)
        else:
            score = accuracy_score(new_labels[val_ind], preds)

        if (epoch % 5 == 0) or (epoch == epochs):
            print('Epoch: [{0}] '
                    'Elapsed {elapsed:s} '
                    'Train Loss: {trn_loss:.4f} '
                    'Valid Loss: {val_loss:.4f} '
                    'Score: {score:.4f} '
                    'LR: {lr:.6f}  '
                    .format(epoch,
                            elapsed=asMinutes(time.time() - start),
                            trn_loss=trn_losses, val_loss=val_losses, score=score,
                            lr=optimizer.param_groups[0]['lr']))
        loss_log.append([trn_losses, val_losses])
        if best_loss > val_losses:
            best_loss = val_losses
            torch.save(model.state_dict(), outdir.joinpath(f'{prefix}_{model_name}_seed{seed}_fold{fold}.pth'))
            n_patient = 0
        else:
            n_patient += 1
            if n_patient > max_patient:
                print(f"Early stopped at Epoch {epoch} | Best Loss: {best_loss:.4f}.")
                break

    # save training history
    # np.save(outdir.joinpath(f"train_log_{model_name}_seed{seed}_fold{fold}"), np.array(loss_log))
    return None



def cross_validation(data, outdir, labels, **kwargs):
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
    
    # Correct labels
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    if not np.all(labels == new_labels):
        pd.DataFrame({"label": le.classes_, "label_for_model": np.arange(len(le.classes_))}).to_csv(outdir.joinpath(f"{prefix}_corresp_table.csv"), index=False)
        print("Convert labels so that they starts with 0 and the correspondence table was saved by name 'corresp_table.csv'.")
    
    # Stratification
    fold_arr = np.full(len(inp_mean), -1)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (trn_ind , val_ind) in enumerate(skf.split(new_labels, new_labels)):
        fold_arr[val_ind] = fold
    
    for fold in range(n_splits):
        print("-"*60)
        print("-"*20, f"Traning Fold {fold}", "-"*20)
        print("-"*60)
        trn_ind = np.where(fold_arr != fold)[0]
        val_ind = np.where(fold_arr == fold)[0]
        train(data, outdir, new_labels, trn_ind, val_ind, fold=fold, **kwargs)
    
    return fold_arr

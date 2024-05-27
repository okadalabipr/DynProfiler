from pathlib import Path
from datetime import datetime
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)

from .utils import seed_everything, AverageMeter, asMinutes
from .models import SimSiam, CNN1d, LSTM, GRU, VanillaRNN
from .datasets import CustomDataset, SupervisedDataset

def set_param(name_, default_, **kwargs):
    param = kwargs.get(name_, None)
    if param is None:
        param = default_
    return param

def embed(data, outdir, **kwargs):
    outdir = Path(outdir)
    if isinstance(data, dict):
        inp_mean = data["mean"]
        if "std" in data.keys():
            inp_std = data["std"]
        else:
            inp_std = None
    else:
        inp_mean = data
        inp_std = None
    
    sampling_aug = set_param("sampling_aug", False, **kwargs)
    if sampling_aug and inp_std is None:
        raise Exception('If sampling_aug is Ture, then data["std"] must not be None.')
    
    prefix = set_param("prefix", datetime.now().strftime("%Y%m%d"), **kwargs)
    
    # trainig setting
    seed = set_param("seed", 42, **kwargs)
    epochs = set_param("epochs", 200, **kwargs)
    lr = float(set_param("lr", 1e-4, **kwargs))
    batch_size = set_param("batch_size", 16, **kwargs)
    num_workers = set_param("num_workers", 2, **kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("GPU device was not found. The training process might get slow.")
    
    # Define dataset
    seed_everything(seed)
    ds = CustomDataset(inp_mean=inp_mean, inp_std=inp_std, train=True, **kwargs)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)
    
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
    model = SimSiam(base_model).to(device)
    model.train()
    
    # Start Train
    criterion = nn.CosineSimilarity(dim=1).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    start = end = time.time()
    loss_log = []
    best_loss = 0.
    for epoch in range(1, epochs+1):
        losses = AverageMeter()
        for step, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            bs = x.size(0)

            p1, p2, z1, z2 = model(x, y)

            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            losses.update(loss.item(), bs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch % 10 == 0) or (epoch == epochs):
            print('Epoch: [{0}] '
                    'Elapsed {elapsed:s} '
                    'Loss: {loss.avg:.4f} '
                    'LR: {lr:.6f}  '
                    .format(epoch,
                            elapsed=asMinutes(time.time() - start),
                            loss=losses,
                            lr=optimizer.param_groups[0]['lr']))
        loss_log.append(losses.avg)
        if best_loss > losses.avg:
            best_loss = losses.avg
            torch.save(model.state_dict(), outdir.joinpath(f'{prefix}_{model_name}_SimSiam_seed{seed}_epochs{epochs}.pth'))
    
    # save training history
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(np.arange(1, epochs+1), loss_log, color="#2b2b2b")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(outdir.joinpath(f"{prefix}_train_log.png"))
    plt.close()
    np.save(outdir.joinpath("train_log"), np.array(loss_log))

    # Embedding
    model.load_state_dict(
        torch.load(outdir.joinpath(f'{prefix}_{model_name}_SimSiam_seed{seed}_epochs{epochs}.pth'))
    )
    model.eval()

    ds = CustomDataset(inp_mean=inp_mean, inp_std=inp_std, train=False, **kwargs)
    dl = DataLoader(ds, batch_size=batch_size*2, shuffle=False,
                    num_workers=num_workers, pin_memory=True, drop_last=False)
    features = []
    for inp in dl:
        inp = inp.to(device)
        with torch.no_grad():
            feat = model.model(inp)
        features.append(feat.detach().to("cpu").numpy())
    features = np.concatenate(features, axis=0)
    np.save(outdir.joinpath(f'{prefix}_{model_name}_SimSiam_seed{seed}_epochs{epochs}'), features)


    

                
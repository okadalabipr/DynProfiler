import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# augmentations
def random_sampling(m, s, sd_scale=1, **kwargs):
    scale = sd_scale * 1000
    eps = random.randint(-scale, scale) / scale
    return m + eps * s

def random_noise(m, min_scale=30, max_scale=100, **kwargs):
    scale = random.uniform(min_scale, max_scale)
    return m + np.random.normal(size=m.shape) / scale
    
def random_shift(m, max_shift=20, **kwargs):
    shift = random.randint(1, max_shift)
    if np.random.rand() < 0.5:
        # right shift
        m = np.concatenate([np.zeros(shape=(m.shape[0], shift)), m[:, :-shift]], axis=1)
    else:
        # left shift
        m = np.concatenate([m[:, shift:], np.zeros(shape=(m.shape[0], shift))], axis=1)
    return m

def random_mask(m, prob=0.2, **kwargs):
    mask = (np.random.random(m.shape) < prob).astype(bool)
    return np.where(mask, 0, m)

def random_flip(m, **kwargs):
    return m[:, ::-1].copy()


class CustomDataset(Dataset):
    def __init__(self, inp_mean, inp_std=None,
                 sampling_aug=True, sampling_aug_prob=0.6,
                 noise_aug=False, noise_aug_prob=0.,
                 shift_aug=False, shift_aug_prob=0.,
                 mask_aug=False, mask_aug_prob=0.,
                 flip_aug=False, flip_aug_prob=0.,
                 train=True, **kwargs):
        self.inp_mean = inp_mean
        self.inp_std = inp_std
        self.sampling_aug = sampling_aug
        self.sampling_aug_prob = sampling_aug_prob
        self.noise_aug = noise_aug
        self.noise_aug_prob = noise_aug_prob
        self.shift_aug = shift_aug
        self.shift_aug_prob = shift_aug_prob
        self.mask_aug = mask_aug
        self.mask_aug_prob = mask_aug_prob
        self.flip_aug = flip_aug
        self.flip_aug_prob = flip_aug_prob
        self.train = train
        self.kwargs = kwargs
        if train and inp_std is None:
            ValueError("'train' is True but 'inp_std' is None")
    
    def __len__(self):
        return len(self.inp_mean)
    
    def __getitem__(self, idx):
        if self.train:
            aug = self.inp_mean[idx].copy()
            # random sampling
            if self.sampling_aug and (np.random.rand() <= self.sampling_aug_prob):
                aug = random_sampling(aug, self.inp_std[idx], **self.kwargs)
            # random noise
            if self.noise_aug and (np.random.rand() <= self.noise_aug_prob):
                aug = random_noise(aug, **self.kwargs)
            # random shift
            if self.shift_aug and (np.random.rand() <= self.shift_aug_prob):
                aug = random_shift(aug, **self.kwargs)
            # random mask
            if self.mask_aug and (np.random.rand() <= self.mask_aug_prob):
                aug = random_mask(aug, **self.kwargs)
            # random flip
            if self.flip_aug and (np.random.rand() <= self.flip_aug_prob):
                aug = random_flip(aug, **self.kwargs)
            
            if self.kwargs.get("name", "1DCNN").lower() not in ["lstm", "gru", "rnn"]:
                return torch.from_numpy(self.inp_mean[idx]).float(), torch.from_numpy(aug).float()
            else:
                return torch.from_numpy(self.inp_mean[idx]).float().permute([1, 0]), torch.from_numpy(aug).float().permute([1, 0])
        
        else:
            if self.kwargs.get("name", "1DCNN").lower() not in ["lstm", "gru", "rnn"]:
                return torch.from_numpy(self.inp_mean[idx]).float()
            else:
                return torch.from_numpy(self.inp_mean[idx]).float().permute([1, 0])

class SupervisedDataset(Dataset):
    def __init__(self, inp_mean, labels=None, train=True, **kwargs):
        self.inp_mean = inp_mean
        self.labels = labels
        self.train = train
        self.kwargs = kwargs
        if train:
            self.label_dtype = torch.float if len(np.unique(labels)) == 2 else torch.long
        if train and labels is None:
            ValueError("'train' is True but 'labels' is None")
    
    def __len__(self):
        return len(self.inp_mean)
    
    def __getitem__(self, idx):
        if self.train:
            labels = torch.tensor(self.labels[idx], dtype=self.label_dtype)
            if self.kwargs.get("name", "1DCNN").lower() not in ["lstm", "gru", "rnn"]:
                return torch.from_numpy(self.inp_mean[idx]).float(), labels
            else:
                return torch.from_numpy(self.inp_mean[idx]).float().permute([1, 0]), labels
        else:
            if self.kwargs.get("name", "1DCNN").lower() not in ["lstm", "gru", "rnn"]:
                return torch.from_numpy(self.inp_mean[idx]).float()
            else:
                return torch.from_numpy(self.inp_mean[idx]).float().permute([1, 0])
        
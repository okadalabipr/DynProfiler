import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import Dataset, DataLoader

class GeM(nn.Module):
    '''
    Code modified from the 2d code in
    https://amaarora.github.io/2020/08/30/gempool.html
    '''
    def __init__(self, kernel_size=8, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool1d(x.clamp(min=eps).pow(p), self.kernel_size).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + \
                ', ' + 'kernel_size=' + str(self.kernel_size) + ')'


class SimSiam(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

    def forward(self, x1, x2):
        z1 = self.model.projector(x1)
        z2 = self.model.projector(x2)

        p1 = self.model.predictor(z1)
        p2 = self.model.predictor(z2)
        return p1, p2, z1.detach(), z2.detach()

class CNN1d(nn.Module):
    def __init__(self, in_feat, emb_size, seq_len, hidden_size=None, **kwargs):
        super().__init__()
        self.in_feat = in_feat
        self.emb_size = emb_size
        self.hidden_size = hidden_size if hidden_size is not None else emb_size // 2
        self.seq_len = seq_len
        self.seq_len_1 = math.floor((seq_len - 2) / 2)

        depth = 0
        seq_len_ = self.seq_len_1
        while seq_len_ >= 3:
            seq_len_ = math.floor((seq_len_ - 2) / 3)
            depth += 1
        self.depth = depth
        

        cnn_blocks = nn.ModuleList()
        cnn_blocks.append(nn.Sequential(
            nn.Conv1d(in_feat, self.hidden_size, 3),
            GeM(kernel_size=2),
            nn.BatchNorm1d(self.hidden_size),
            nn.SiLU()
        ))
        
        for i in range(2, self.depth+2):
            if i != self.depth+1:
                cnn_blocks.append(nn.Sequential(
                    nn.Conv1d(self.hidden_size, self.emb_size, 3),
                    GeM(kernel_size=3),
                    nn.BatchNorm1d(self.emb_size),
                    nn.SiLU(),
                    nn.Conv1d(self.emb_size, self.hidden_size, 1),
                    nn.BatchNorm1d(self.hidden_size),
                    nn.SiLU()
                ))
            else:
                cnn_blocks.append(nn.Sequential(
                    nn.Conv1d(self.hidden_size, self.emb_size, 3),
                    GeM(kernel_size=3),
                    nn.BatchNorm1d(self.emb_size)
                ))
        self.cnn_blocks = cnn_blocks

        self.predictor = nn.Linear(self.emb_size, self.emb_size)

    def projector(self, x):
        for block in self.cnn_blocks:
            x = block(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.projector(x)
        x = self.predictor(x)
        return x


class LSTM(nn.Module):
    def __init__(self, in_feat, emb_size, hidden_size=None, num_layers=1, dropout=0.2, **kwargs):
        super().__init__()
        self.in_feat = in_feat
        self.emb_size = emb_size
        self.hidden_size = hidden_size if hidden_size is not None else emb_size // 2
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_feat, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.bn = nn.BatchNorm1d(self.emb_size)
        self.fc = nn.Linear(self.emb_size, self.emb_size)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
    
    def projector(self, x):
        x, (_, _) = self.lstm(x)
        x = self.bn(self.fc(x[:, -1, :]))
    
    def forward(self, x):
        x = self.projector(x)
        x = self.predictor(x)
        return x
    

class GRU(nn.Module):
    def __init__(self, in_feat, emb_size, hidden_size=None, num_layers=1, dropout=0.2, **kwargs):
        super().__init__()
        self.in_feat = in_feat
        self.emb_size = emb_size
        self.hidden_size = hidden_size if hidden_size is not None else emb_size // 2
        self.num_layers = num_layers
        self.gru = nn.GRU(in_feat, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        self.bn = nn.BatchNorm1d(self.emb_size)
        self.fc = nn.Linear(self.emb_size, self.emb_size)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
    
    def projector(self, x):
        x, _ = self.gru(x)
        x = self.bn(self.fc(x[:, -1, :]))
    
    def forward(self, x):
        x = self.projector(x)
        x = self.predictor(x)
        return x

class VanillaRNN(nn.Module):
    def __init__(self, in_feat, emb_size, hidden_size=None, num_layers=2, dropout=0.2, **kwargs):
        super().__init__()
        self.in_feat = in_feat
        self.emb_size = emb_size
        self.hidden_size = hidden_size if hidden_size is not None else emb_size // 2
        self.num_layers = num_layers
        self.rnn = nn.RNN(in_feat, self.hidden_size, self.num_layers, nonlinearity='relu', batch_first=True, dropout=dropout, bidirectional=True)

        self.bn = nn.BatchNorm1d(self.emb_size)
        self.fc = nn.Linear(self.emb_size, self.emb_size)
        self.predictor = nn.Linear(self.emb_size, self.emb_size)
    
    def projector(self, x):
        x, _ = self.rnn(x)
        x = self.bn(self.fc(x[:, -1, :]))
    
    def forward(self, x):
        x = self.projector(x)
        x = self.predictor(x)
        return x

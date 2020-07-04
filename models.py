import torch
import torch.nn as nn

import numpy as np

__all__ = ['DenseEmbeddingNet', 'ResDenseEmbeddingNet']

class DenseEmbeddingNet(nn.Module):
    # code from fast.ai tabular model
    # https://github.com/fastai/fastai/blob/3b7c453cfa3845c6ffc496dd4043c07f3919270e/fastai/tabular/models.py#L6
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super().__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz
        self.n_embs = len(emb_szs) - 1
        if ps == 0:
          ps = np.zeros(self.n_embs)
        # input layer
        layers = [nn.Linear(self.in_sz, emb_szs[0]),
                  actn]
        # hidden layers
        for i in range(self.n_embs):
            layers += self.bn_drop_lin(n_in=emb_szs[i], n_out=emb_szs[i+1], bn=use_bn, p=ps[i], actn=actn)
        # output layer
        layers.append(nn.Linear(emb_szs[-1], self.out_sz))
        self.fc = nn.Sequential(*layers)
        
    def bn_drop_lin(self, n_in:int, n_out:int, bn:bool=True, p:float=0., actn:nn.Module=None):
        "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None: layers.append(actn)
        return layers
              
    def forward(self, x):
        output = self.fc(x)
        return output


def make_fc_unit(in_sz, out_sz, bn=True, p=0., actn=nn.ReLU()):
    layers = [nn.Linear(in_sz, out_sz)]
    if bn: layers.append(nn.BatchNorm1d(out_sz))
    if actn is not None: layers.append(actn)
    if p != 0: layers.append(nn.Dropout(p))
    return layers


class BasicBlock(nn.Module):
    
    def __init__(self, in_sz, hidden_sz, out_sz, bn=True, p=0., actn=nn.ReLU()):
        super().__init__()
        layers = make_fc_unit(in_sz, hidden_sz, bn=True, p=0., actn=nn.ReLU())
        layers += make_fc_unit(hidden_sz, out_sz, bn=True, p=0., actn=nn.ReLU())

        self.FC = nn.Sequential(*layers)
        
        shortcut_layers = make_fc_unit(in_sz, out_sz, bn=True, p=0., actn=nn.ReLU())
        self.shortcut = nn.Sequential(*shortcut_layers)
        
    def forward(self, x):
        out = self.FC(x) + self.shortcut(x)
        return out

    
class ResDenseEmbeddingNet(nn.Module):
    def __init__(self, in_sz, out_sz, emb_szs, ps, use_bn=True, actn=nn.ReLU()):
        super().__init__()
        self.n_embs = len(emb_szs) - 1
        if ps == 0: ps = np.zeros(self.n_embs)
        # input FC_unit
        layers = make_fc_unit(in_sz, emb_szs[0], bn=True, p=0., actn=nn.ReLU())
        # hidden block
        for i in range(0, self.n_embs-1, 2):
            layers.append(BasicBlock(in_sz=emb_szs[i], 
                                 hidden_sz=emb_szs[i+1], 
                                 out_sz=emb_szs[i+2],
                                 bn=True, p=0., actn=nn.ReLU())
                         )
        # output layer
        if self.n_embs % 2 != 0:
            layers += make_fc_unit(emb_szs[-2], emb_szs[-1], bn=True, p=0., actn=nn.ReLU())
        layers += make_fc_unit(emb_szs[-1], out_sz, bn=True, p=0., actn=nn.ReLU())
        self.ResDense = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.ResDense(x)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest
import copy

# copies don't reference same memory
def clones(module, n):
    return nn.ModuleList([copy.deepcoy(module) for _ in range(n)])

class LayerNorm(nn.Module):
    def __init__(self, features, epsilon = 1e-8):
        super().__init__()
        self.mults = nn.Parameter(torch.ones(features))
        self.adds = nn.Parameter(torch.zeros(features))
        self.epsilon = epsilon
    def forward(self, x):
        m = x.mean(-1, keepdim = True)
        v = x.var(-1, keepdim = True)
        x = x - m / (v + self.epsilon).sqrt()
        return x*self.mults + self.adds

# Mask future values for 'causality' flag
def attention(query, key, value, dropout=None):
    # q,k,v are all of size d_k = d_v
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

    # attention weights
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0):
        super().__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.attn = None

    def forward(self, query, key, value):
        nbatches = query.shape[0]
        # broadcast could train on same weights
        lx = zip_longest([self.lin], (query, key, value), fillvalue = self.lin)  # [(lin,q), (lin,k), (lin, v)] with broadcast
        q, k, v = [layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for layer, x in lx]

        x, self.attn = attention(q, k, v, dropout = self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.lin(x)
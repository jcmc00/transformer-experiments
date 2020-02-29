import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest

class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-8):
        super().__init__()
        self.mults = nn.Parameter(torch.ones(features))
        self.adds = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        m = x.mean(-1, keepdim = True)
        v = x.var(-1, keepdim = True)
        x = x - m / (v + self.epsilon).sqrt()
        return x*self.mults + self.adds

def attention(query, key, value, mask = None, dropout= None):
    # q,k,v are all of size d_k = d_v
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(d_k)

    if mask:
        score = scores.masked_fill(mask == 0, -1e9)
    
    # attention weights
    p_attn = F.softmax(scores, dim = -1)
    if dropout.p > 0:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super().__init__()
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.lin = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.attn = None

    def forward(self, query, key, value, mask = None):
        if mask:
            mask = mask.unsqueeze(1)

        nbatches = query.shape[0]
        # broadcast could train on same weights
        lx = zip_longest([self.lin], (query, key, value), fillvalue = self.lin)  # [(lin,q), (lin,k), (lin, v)] with broadcast
        q, k, v = [layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for layer, x in lx]

        x, self.attn = attention(q, k, v, dropout = self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.lin(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
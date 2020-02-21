import torch
import torch.nn as nn
import torch.nn.functional as F

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
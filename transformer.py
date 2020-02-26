import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
import copy

# copies don't reference same memory
def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadedAttention(d_model, n_heads, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model, epsilon=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        input_norm = self.layer_norm(x) # LayerNorm before attn
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + x
        return self.feed_forward(out)
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LayerNorm, MultiHeadedAttention, PositionwiseFeedForward
import copy

def clones(module, n):
    return nn.ModuleList([module for _ in range(n)])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout = 0.1):
        super().__init__()
        self.self_attn = MultiHeadedAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = LayerNorm(d_model, epsilon=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        input_norm = self.layer_norm(x) # LayerNorm before attn
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask = mask)
        out = self.dropout(context) + x
        return self.feed_forward(out)

class TransformerEncoder(EncoderBase):

    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout, embeddings):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = clones(TransformerEncoder(d_model, n_heads, d_ff, dropout), num_layers)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, lengths=None):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)
        # Run the forward pass of every layer of the tranformer.
        for i in range(self.num_layers):
            out = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths
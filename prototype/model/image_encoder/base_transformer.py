from collections import OrderedDict

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

global LAYER_NORM 
LAYER_NORM = True

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        if LAYER_NORM:
            ret = super().forward(x)
        else:
            ret = x
        return ret
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, dropout: float = 0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            # ("dropout_1", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            # ("dropout_2", nn.Dropout(dropout))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(
            dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, checkpoint: bool = False, dropout: float = 0., emb_dropout: float = 0.):
        super().__init__()
        self.width = width
        self.layers = layers
        self.checkpoint = checkpoint
        self.dropout = nn.Dropout(emb_dropout)
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, dropout=dropout) for _ in range(layers)])

    def checkpoint_fwd(self, layer, input, segments=2):
        """checkpoint forward"""
        # Make sure that the input to checkpoint have requires_grad=True, so that
        # the autograd can take care of the checkpointed part of model
        if not input.requires_grad:
            input = input.detach()
            input.requires_grad = True
        return checkpoint_sequential(layer, segments, input)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        if self.checkpoint:
            return self.checkpoint_fwd(self.resblocks, x, self.layers)
        return self.resblocks(x)


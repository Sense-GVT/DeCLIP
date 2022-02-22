import torch
import torch.nn.functional as F
from torch import nn
from .base_transformer import Transformer, LayerNorm

class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, embed_dim: int, checkpoint: bool, dropout: float=0, emb_dropout: float=0):
        super().__init__()
        self.input_resolution = input_resolution
        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True
        # self.freeze_conv1 = False
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, dropout=dropout, emb_dropout=emb_dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:
            print('-----------------------------------------------------------')
            for layer in [self.conv1]:
                print('set conv1.requires_grad to False')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self


    def forward(self, x: torch.Tensor, return_dense=False, return_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        dense_feat = x[:, 1:, :]
        x = self.ln_post(x[:, 0, :])
        feature = x

        if self.proj is not None:
            x = x @ self.proj

        ret = [x]
        if return_dense:
            ret.append(dense_feat)
        if return_feature:
            ret.append(feature)
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)
        # if return_dense:
        #     return x, dense_feat

        # return x

def visual_transformer_B32(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads, 
        'input_resolution': 224,
        'patch_size': 32,
        'width': vision_width,
        'checkpoint': False
    }
    default_kwargs.update(**kwargs)
    model = VisualTransformer(**default_kwargs)
    return model

def visual_transformer_B16(**kwargs):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    default_kwargs = {
        # 'output_dim': 512, from config
        'layers':vision_layers,
        'heads': vision_heads,
        'input_resolution': 224,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': False
    }
    default_kwargs.update(**kwargs)
    model = VisualTransformer(**default_kwargs)
    return model

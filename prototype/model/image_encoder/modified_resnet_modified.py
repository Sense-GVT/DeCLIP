from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import linklink as link
from linklink.nn import SyncBatchNorm2d
from linklink.nn import syncbnVarMode_t

BN = None

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = BN(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = BN(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = BN(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", BN(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(
            spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

def simple_group_split(world_size, rank, num_groups):
    groups = []
    rank_list = np.split(np.arange(world_size), num_groups)
    rank_list = [list(map(int, x)) for x in rank_list]
    for i in range(num_groups):
        groups.append(link.new_group(rank_list[i]))
    group_size = world_size // num_groups
    return groups[rank // group_size]

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, embed_dim, heads, input_resolution=224, width=64,
                 bn_group_size=1,
                 bn_var_mode=syncbnVarMode_t.L2,
                 bn_sync_stats=False,
                 use_sync_bn=True,
                 fc_embed=False):

        global BN

        rank = link.get_rank()
        world_size = link.get_world_size()
        bn_group = simple_group_split(world_size, rank, world_size // bn_group_size)

        def BNFunc(*args, **kwargs):
            return SyncBatchNorm2d(*args, **kwargs,
                                   group=bn_group,
                                   sync_stats=bn_sync_stats,
                                   var_mode=bn_var_mode)


        if use_sync_bn:
            BN = BNFunc
            print('[Rank {}] use SyncBatchNorm in bn_group {}'.format(rank, bn_group), flush=True)
        else:
            BN = nn.BatchNorm2d

        super().__init__()
        output_dim = embed_dim
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = BN(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = BN(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = BN(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads)
        self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)

        std = self.attnpool.c_proj.in_features ** -0.5
        nn.init.normal_(self.attnpool.q_proj.weight, std=std)
        nn.init.normal_(self.attnpool.k_proj.weight, std=std)
        nn.init.normal_(self.attnpool.v_proj.weight, std=std)
        nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)


    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_dense=False, return_feature=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        dense = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        if x.size(3) == 7:
            x = self.attnpool(x)
            feature = x
            x = self.fc(x)
        else:
            x = self.adaptivepool(x).squeeze()
            feature = x
            x = self.fc(x)
        ret = [x]
        if return_dense:
            ret.append(dense)
        if return_feature:
            ret.append(feature)
        if len(ret) == 1:
            return ret[0]
        return tuple(ret)

def modified_resnet_R50(**kwargs):
    vision_width = 64
    vision_heads = vision_width * 32 // 64
    default_kwargs = {
        # 'output_dim': 1024, from config
        'layers':(3, 4, 6, 3),
        'heads': vision_heads,
        'input_resolution': 224,
        'width': vision_width
    }
    default_kwargs.update(**kwargs)
    model = ModifiedResNet(**default_kwargs)
    return model

def modified_resnet_R101(**kwargs):
    vision_width = 64
    vision_heads = vision_width * 32 // 64
    default_kwargs = {
        # 'output_dim': 1024, from config
        'layers':(3, 4, 23, 3),
        'heads': vision_heads,
        'input_resolution': 224,
        'width': vision_width
    }
    default_kwargs.update(**kwargs)
    model = ModifiedResNet(**default_kwargs)
    return model


from collections import OrderedDict
from socket import IP_DEFAULT_MULTICAST_LOOP
from typing import Tuple, Union, List
import ipdb
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

import timm
from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
from .image_encoder.modified_resnet import modified_resnet_R50, modified_resnet_R101
from .text_encoder.text_transformer import text_transformers

import linklink as link
from linklink.nn import SyncBatchNorm2d
from linklink.nn import syncbnVarMode_t


BN = None

__all__ = ['clip_res50', 'clip_vitb32']

class AllGather(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = link.get_rank()
        ctx.world_size = link.get_world_size()

#         y = tensor.new(ctx.world_size, *tensor.size())
        
        y = [tensor.new(*tensor.size()) for _ in range(ctx.world_size)]
        
        link.allgather(y, tensor)

        y = torch.cat(y, 0).view(-1, *tensor.size())
        
        return y

    @staticmethod
    def backward(ctx, grad_output):
        in_grad = torch.zeros_like(grad_output)
        in_grad.copy_(grad_output)
        # sum grad for gathered tensor
        link.allreduce(in_grad)
        # split
        return in_grad[ctx.rank]

    

class CLIP(nn.Module):
    def __init__(self,image_encode, text_encode, use_allgather):
        super().__init__()
        self.use_allgather = use_allgather
        self.visual =image_encode
        self.encode_text = text_encode
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1/0.07))
        #nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def text_parameters(self):
        param = [self.logit_scale]
        if self.encode_text.text_encode_type == 'Transformer':
            param.append(self.encode_text.positional_embedding)
        elif self.encode_text.text_encode_type == 'Bert':
            # print('Bert', self.encode_text.text_transformer.cls.predictions, flush=True)
            # param.extend([self.encode_text.text_transformer.cls.predictions.decoder.weight,
            #               self.encode_text.text_transformer.cls.predictions.bias])
            param.extend([self.encode_text.text_transformer.cls.predictions.bias])
        return param

    def text_modules(self):
        if self.encode_text.text_encode_type == 'Transformer':
            return [self.encode_text.transformer, self.encode_text.text_projection, self.encode_text.token_embedding, self.encode_text.ln_final]
        elif self.encode_text.text_encode_type == 'Bert':
            # print('Bert', self.encode_text.text_transformer, flush=True)
            return [self.encode_text.text_transformer.bert, self.encode_text.text_projection,
                    self.encode_text.text_transformer.cls.predictions.transform]
                    # self.encode_text.text_transformer.cls.predictions.decoder,  # decoder: bias
        else:
            import ipdb
            ipdb.set_trace()
            return [self.encode_text.text_transformer, self.encode_text.text_projection]

    def visual_parameters(self):
        return []

    def visual_modules(self):
        return [self.visual]

    @property
    def dtype(self):
        try:
            return self.visual.conv1.weight.dtype
        except:
            try:
                return self.visual.head.weight.dtype
            except:
                try:
                    return self.visual.stem[0].weight.dtype
                except:
                    return self.encode_text.text_projection.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def sample_captions(self, texts):
        return [text[0] for text in texts]

    def all_gather(self, input):
        output = AllGather.apply(input)
        output = output.view(-1, *(output.shape[2:]))
        return output

    def forward(self, input, all_gather=False):
        # input
        images = input['images']
        texts = input['captions']
        texts = self.sample_captions(texts)
        # text&image encode 
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)


        # normalized features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale.data = torch.clamp(logit_scale.data, max=100)

        if self.training and self.use_allgather or all_gather:
            gathered_image_features = self.all_gather(image_features)
            gathered_text_features = self.all_gather(text_features)

            logits_per_image = logit_scale * image_features @ gathered_text_features.t()
            logits_per_text = logit_scale * text_features @ gathered_image_features.t()
        else:
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logit_scale * text_features @ image_features.t()

        return logits_per_image, logits_per_text


def clip_res50(**kwargs): 
    """
    Constructs a clip_res50 model.
    """
    image_encode = modified_resnet_R50(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = CLIP(image_encode,text_encode,**kwargs['clip'])
    return model

def clip_vitb32(**kwargs):
    """'
    Constructs a clip_ViT_B32 model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = CLIP(image_encode,text_encode,**kwargs['clip'])
    return model

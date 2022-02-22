from collections import OrderedDict
from socket import IP_DEFAULT_MULTICAST_LOOP
from typing import Tuple, Union, List
import ipdb
import timm
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os

from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
from .image_encoder.modified_resnet import modified_resnet_R50, modified_resnet_R101
from .text_encoder.text_transformer import text_transformers
import linklink as link
from linklink.nn import SyncBatchNorm2d #, SyncBatchNorm1d
from linklink.nn import syncbnVarMode_t
from random import choice


# BN = None
BN = SyncBatchNorm2d

__all__ = ['slip_res50', 'slip_vitb32']


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


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=1024, num_layers=3, out_bn=True):
        super(projection_MLP, self).__init__()
        ''' page 3 baseline setting
        Projection MLP. The projection MLP (in f) has BN ap-
        plied to each fully-connected (fc) layer, including its out-
        put fc. Its output fc has no ReLU. The hidden fc is 2048-d.
        This MLP has 3 layers.
        '''
        self.num_layers = num_layers

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(in_dim, hidden_dim)
        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn1 = BN(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = BN(hidden_dim)

        if self.num_layers == 3:
            self.relu2 = nn.ReLU(inplace=True)
            self.linear3 = nn.Linear(hidden_dim, out_dim)
            self.bn3 = nn.BatchNorm1d(hidden_dim)
            # self.bn3 = BN(hidden_dim)
            self.out_bn=out_bn

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        # b, _ = x.shape
        # layer 1
        x = self.linear1(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn1(x)
        x = self.relu1(x)
        # x.reshape(b, self.hidden_dim)

        # layer 2
        x = self.linear2(x)
        # x.reshape(b, self.hidden_dim, 1)
        x = self.bn2(x)


        if self.num_layers == 3:
            x = self.relu2(x)
            # x.reshape(b, self.hidden_dim)
            # layer 3
            x = self.linear3(x)
            # x.reshape(b, self.out_dim, 1)
            # x.reshape(b, self.out_dim)
            if self.out_bn:
                x = self.bn3(x)

        return x


class CLIP(nn.Module):
    def __init__(self,image_encode, text_encode, use_allgather):
        super().__init__()
        self.use_allgather = use_allgather
        self.visual =image_encode
        self.text_encoder = text_encode
        self.logit_scale = nn.Parameter(torch.ones([1]))
        # self.logit_scale = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale, np.log(1/0.07))
        #nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def text_parameters(self):
        param = [self.logit_scale]
        if self.text_encoder.text_encode_type == 'Transformer':
            param.append(self.text_encoder.positional_embedding)
        elif self.text_encoder.text_encode_type == 'Bert':
            # print('Bert', self.text_encoder.text_transformer.cls.predictions, flush=True)
            # param.extend([self.text_encoder.text_transformer.cls.predictions.decoder.weight,
            #               self.text_encoder.text_transformer.cls.predictions.bias])
            param.extend([self.text_encoder.text_transformer.cls.predictions.bias])
        return param

    def text_modules(self):
        if self.text_encoder.text_encode_type == 'Transformer':
            return [self.text_encoder.transformer, self.text_encoder.text_projection, self.text_encoder.token_embedding, self.text_encoder.ln_final]
        elif self.text_encoder.text_encode_type == 'Bert':
            # print('Bert', self.text_encoder.text_transformer, flush=True)
            return [self.text_encoder.text_transformer.bert, self.text_encoder.text_projection,
                    self.text_encoder.text_transformer.cls.predictions.transform]
                    # self.text_encoder.text_transformer.cls.predictions.decoder,  # decoder: bias
        else:
            import ipdb
            ipdb.set_trace()
            return [self.text_encoder.text_transformer, self.text_encoder.text_projection]

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
                    return self.text_encoder.text_projection.weight.dtype

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
        text_features = self.text_encoder(texts)

        # print('feature mean', image_features.mean(-1), 'abs mean', image_features.abs().mean(-1), 'abs max', image_features.abs().max(), 'std', image_features.std(-1), flush=True)

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


class SLIP(CLIP):
    def __init__(self,image_encode, text_encode, use_allgather,
                 EDA=True, feature_dim=1024, sim_dim=256, forward_type='split', return_sim=False):
        super(SLIP, self).__init__(image_encode, text_encode, use_allgather)
        self.return_sim = return_sim
        if self.return_sim:
            self.predictor_sim = projection_MLP(feature_dim, hidden_dim=4096, out_dim=sim_dim, out_bn=False)
        self.forward_type = forward_type

    def text_modules(self):
        ret = super(SLIP, self).text_modules()
        return ret

    def visual_modules(self):
        ret = super(SLIP, self).visual_modules()
        ret.extend([self.predictor_sim])
        return ret

    def encode_image(self, image, return_dense=False, return_sim=False):
        if return_dense:
            output = self.visual(image.type(self.dtype), return_dense=return_dense, return_feature=return_sim)
        else:
            output = self.visual(image.type(self.dtype), return_feature=return_sim)
        if return_sim:
            sim_out = self.predictor_sim(output[-1])
            output = (*output[:-1], sim_out)  # change image feature
        return output

    def encode_text(self, text, text_mask_type=None, return_sim=False):
        assert not return_sim
        if text_mask_type:
            output = self.text_encoder(text, mask_type=text_mask_type)
        else:
            output = self.text_encoder(text)
        return output

    def forward(self, input, return_dict=False):
        # input
        images = input['images']
        images_base, images_1, images_2 = torch.split(images, [3,3,3], dim=1)
        texts = input['captions']
        texts = self.sample_captions(texts)
        text_features = self.encode_text(texts)

        image_features = self.encode_image(images_base)
        image_features_1, image_sim_1 = self.encode_image(images_1, return_sim=True)
        image_features_2, image_sim_2 = self.encode_image(images_2, return_sim=True)

        # normalized features
        image_features = image_features / (image_features.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)

        # image_sim_1 = image_sim_1 / (image_sim_1.norm(dim=-1, keepdim=True))
        # image_sim_2 = image_sim_2 / (image_sim_2.norm(dim=-1, keepdim=True))

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        if self.training and self.use_allgather:
            link.barrier()
            gathered_image_features = self.all_gather(image_features)
            gathered_text_features = self.all_gather(text_features)

            gathered_image_sim_1 = self.all_gather(image_sim_1)
            gathered_image_sim_2 = self.all_gather(image_sim_2)

            logits_per_image = logit_scale * image_features @ gathered_text_features.t()
            logits_per_text = logit_scale * text_features @ gathered_image_features.t()
        else:
            raise NotImplementedError('2-View: Not Implemented')

        if return_dict:
            link.barrier()
            ret_dict = {}
            ret_dict['logits'] = logits_per_image, logits_per_text
            ret_dict['sim_features'] = image_sim_1, gathered_image_sim_1, image_sim_2, gathered_image_sim_2
            ret_dict['features'] = text_features, image_features
            return ret_dict
        raise NotImplementedError('Must Return A Dict')


def slip_res50(**kwargs):
    """
    Constructs a slip_res50 model.
    """
    image_encode = modified_resnet_R50(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = SLIP(image_encode,text_encode,**kwargs['clip'])
    return model


def slip_vitb32(**kwargs):
    """
    Constructs a slip_vitb32 model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = SLIP(image_encode,text_encode,**kwargs['clip'])
    return model


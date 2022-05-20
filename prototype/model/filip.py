from collections import OrderedDict
from socket import IP_DEFAULT_MULTICAST_LOOP
from typing import Tuple, Union, List
import ipdb
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os


from .image_encoder.visual_transformer import visual_transformer_B32, visual_transformer_B16
from .image_encoder.modified_resnet import modified_resnet_R50, modified_resnet_R101
from .text_encoder.text_transformer import text_transformers
import linklink as link
from linklink.nn import SyncBatchNorm2d
from linklink.nn import syncbnVarMode_t
from random import choice

from .clip import CLIP


BN = None

__all__ = ['filip_res50', 'filip_vitb32']

class FILIP(CLIP):
    def __init__(self,image_encode, text_encode, use_allgather, nn_size=2**16, nn_topk=1, \
                 return_dense=False, return_caption=False, return_nn_bank=False, text_mask_type=None, \
                 EDA=True, feature_dim=1024, embed_dim=768, forward_type='split', dense_mapping_image=2048, \
                 dense_mapping_language=512, dense_embed_dim=256, mask_rate=0.75, patch_number=14, \
                 text_mae_feature=False, return_simsiam=False, two_view=False, sparse=False, select_topk=False):
        super(FILIP, self).__init__(image_encode, text_encode, use_allgather)
        self.return_dense = return_dense
        self.return_caption = return_caption

        self.text_mask_type = text_mask_type
        self.select_topk = select_topk
        if self.return_dense:
            self.image_mapping = nn.Linear(dense_mapping_image, dense_embed_dim)
            self.text_mapping = nn.Linear(dense_mapping_language, dense_embed_dim)

        self.logit_scale_dense = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_dense, np.log(1/0.07))

        if self.encode_text.text_encode_type == 'Transformer':
            self.sos_index = self.encode_text.tokenizer.encoder["<|startoftext|>"]
            self.padding_idx = 0
        if self.return_caption:
            self.caption_module = TransformerDecoderTextualHead(visual_feature_size=2048, vocab_size=self.encode_text.vocab_size, padding_idx=self.padding_idx)
        else:
            self.caption_module = None
        if text_mask_type is not None:
            enc_dim = self.encode_text.text_projection.weight.shape[-1]
            self.text_label_predictor = nn.Linear(enc_dim, self.encode_text.vocab_size)

    def encode_text_dense(self, texts, return_dense=True):
        text_features, word_features = self.encode_text(texts, return_dense=return_dense)
        word_features_d = self.text_mapping(word_features)
        return word_features_d

    def encode_image_dense(self, image):
        image_features, image_features_dense = self.visual(image.type(self.dtype), return_dense=True)
        image_features_dense = self.image_mapping(image_features_dense)
        return image_features_dense

    def encode_image(self, image, return_all=False):
        output = self.visual(image.type(self.dtype), return_dense=return_all)
        return output

    def get_weighted_dense_logits(self, dense_feat_1, dense_feat_2, top_k=16):
        dense_feat_1 = dense_feat_1 / dense_feat_1.norm(dim=-1, keepdim=True)
        dense_feat_2 = dense_feat_2 / dense_feat_2.norm(dim=-1, keepdim=True)

        logit_scale_dense = self.logit_scale_dense.exp()


        if self.select_topk:
            dense_feat_cross_logit = torch.matmul(dense_feat_1, dense_feat_2.permute(0, 2, 1))
            _, dense_id_1 = torch.topk(dense_feat_cross_logit.sum(dim=2), dim=1, k=top_k)
            _, dense_id_2 = torch.topk(dense_feat_cross_logit.sum(dim=1), dim=1, k=top_k)
            bs, n1 = dense_feat_1.shape[:2]
            dense_id_1 = dense_id_1 + (torch.arange(bs) * n1).to(dense_id_1.device)[:, None]
            selected_feat_1 = dense_feat_1.reshape(bs * n1, -1)[dense_id_1].reshape(bs, top_k, -1)
            bs, n2 = dense_feat_2.shape[:2]
            dense_id_2 = dense_id_2 + (torch.arange(bs) * n2).to(dense_id_2.device)[:, None]
            selected_feat_2 = dense_feat_2.reshape(bs * n2, -1)[dense_id_2].reshape(bs, top_k, -1)

#         selected_feat_1 = dense_feat_1
#         selected_feat_2 = dense_feat_2

        selected_feat_1 = self.all_gather(selected_feat_1)
        selected_feat_2 = self.all_gather(selected_feat_2)


        def get_logits(dense_feat_1, selected_feat_2):
            i, j, k = dense_feat_1.shape
            l, m, k = selected_feat_2.shape
            dense_feat_1 = dense_feat_1.reshape(-1, k)
            selected_feat_2 = selected_feat_2.reshape(-1, k)
            final_logits_1 = logit_scale_dense * dense_feat_1 @ selected_feat_2.t()
            final_logits_1 = final_logits_1.reshape(i, j, l, m).permute(0,2,1,3)
            return final_logits_1
        final_logits_1 = get_logits(dense_feat_1, selected_feat_2).max(dim=-1)[0].mean(dim=-1)
        final_logits_2 = get_logits(dense_feat_2, selected_feat_1).max(dim=-1)[0].mean(dim=-1)
        return final_logits_1, final_logits_2


    def forward(self, input, return_dict=False):
        # data input
        images = input['images']
        images_1, _ = torch.split(images, [3,3], dim=1)
        texts = input['captions']
        texts = self.sample_captions(texts)
        # text
        text_features, word_features, text_labels = self.encode_text(texts, mask_type = self.text_mask_type)
        # image
        image_concat = images_1
        image_features_1, image_features_d = self.encode_image(image_concat, return_all=True)
        # clip
        logit_scale = self.logit_scale.exp()
        image_features_1 = image_features_1 / (image_features_1.norm(dim=-1, keepdim=True))
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True)+1e-10)
        if self.training and self.use_allgather:
            link.barrier()
            gathered_image_features_1 = self.all_gather(image_features_1)
            gathered_text_features = self.all_gather(text_features)
            logits_per_image_1 = logit_scale * image_features_1 @ gathered_text_features.t()
            logits_per_text_1 = logit_scale * text_features @ gathered_image_features_1.t()    
        # filip
        if self.return_dense:
            image_features_d1 = image_features_d
            image_features_d1 = self.image_mapping(image_features_d1)
            word_features_d = self.text_mapping(word_features)
            logits_per_image_dense_1, logits_per_text_dense_1 = self.get_weighted_dense_logits(image_features_d1, word_features_d)
        if return_dict:
            ret_dict = {}
            ret_dict['logits'] = logits_per_image_1, logits_per_text_1
            if self.return_dense:
                ret_dict['dense_logits'] = logits_per_image_dense_1, logits_per_text_dense_1
            return ret_dict
        raise NotImplementedError()



def filip_res50(**kwargs):
    """'
    Constructs a vit mae model.
    """
    image_encode = modified_resnet_R50(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = FILIP(image_encode,text_encode,**kwargs['clip'], dense_mapping_image=2048)
    return model


def filip_vitb32(**kwargs):
    """'
    Constructs a vit mae model.
    """
    image_encode = visual_transformer_B32(**kwargs['image_encode'])
    text_encode = text_transformers(**kwargs['text_encode'])
    model = FILIP(image_encode,text_encode,**kwargs['clip'], dense_mapping_image=768)
    return model

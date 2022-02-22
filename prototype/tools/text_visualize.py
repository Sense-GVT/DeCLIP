from prototype.utils.misc import parse_config, load_state_model
from prototype.utils.dist import  link_dist, DistModule
from prototype.model import model_entry

import torch

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
import os

def load_texts(num):
    caption_path = '/mnt/lustre/share/liujie4/data/clip/Conceptual_Captions_3.3M/dataset_prototype_format.json'
    f = open(caption_path)
    lines = f.readlines()
    captions = []
    for line in lines[:num]:
        meta = json.loads(line)
        captions.append(meta['caption'])
    return captions

# load model
def load_model():
    exp_path = '/mnt/lustre/lichuming/clip/model_zoo_exp/clip_experiments/Conceptual_Captions/clip_res50_adamw_sgd/'
    pretrain_path = '/mnt/lustre/share/liujie4/to_chuming/clip/clip_res50_5M/ckpt.pth.tar'
    config = parse_config(exp_path + 'config.yaml')
    state = torch.load(pretrain_path, 'cpu')
    model = model_entry(config.model)
    model = DistModule(model)
    model.load_state_dict(state['model'], strict=False)

    state_keys = set(state['model'].keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        print(f'missing key: {k}')

    model.cuda()
    model.eval()
    model = model.module
    return model

@torch.no_grad()
def vis(model, captions, dir):

    # calculate attention weights
    trans_self_attn_weights = []
    hooks = []
    layer_num = len(model.transformer.resblocks)
    for i in range(layer_num):
        hooks.append(
            model.transformer.resblocks[i].attn.register_forward_hook(
            lambda self, input, output: trans_self_attn_weights.append(output[1]))
            )

    model.encode_text(captions)

    for hook in hooks:
        hook.remove()

    # [[bs, L, S]] -> [L, S]
    for i, text in enumerate(captions):
        tokens = model.tokenizer.encode(text)
        words = ['start'] + [model.tokenizer.decoder[token][:-4] for token in tokens] + ['end']
        print('visualize caption {}'.format(i))
        renorm = torch.arange(len(words)).view(-1, 1) + 1
        for j in range(layer_num):
            trans_self_attn_weight = trans_self_attn_weights[j][i][:len(words),:len(words)].cpu() * renorm
            trans_self_attn_weight = trans_self_attn_weight.numpy()

            aa = plt.imshow(trans_self_attn_weight, plt.get_cmap('RdBu'))
            plt.colorbar(aa)
            plt.xticks(list(range(len(words))), words, color='blue',rotation=60)
            plt.yticks(list(range(len(words))), words, color='blue',rotation=60)
            plt.axis([-0.5, len(words)-0.5, -0.5, len(words)-0.5])
            filename = 'text_vis_id_{0:0>3}'.format(i) + '_layer_{0:0>2}'.format(j) + '.jpg'
            plt.savefig(dir + '/' + filename, dpi=200)
            plt.close('all')

@link_dist
def main():
    text = 'a dog is looking at a cat eating fish'
    dir = './text_vis'
    os.makedirs(dir, exist_ok=True)
    model = load_model()
    captions = load_texts(100)
    vis(model, captions, dir)

if __name__ == '__main__':
    main()

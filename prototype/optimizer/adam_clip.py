from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW


class AdamWithClip(Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_norm=None, norm_type=2):
        super(AdamWithClip, self).__init__(
            params, lr, betas, eps, weight_decay, amsgrad)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def step(self, closure=None):
        if self.max_norm is not None:
            for group in self.param_groups:
                clip_grad_norm_(group['params'], self.max_norm, self.norm_type)
        super(AdamWithClip, self).step(closure)


class AdamWWithClip(AdamW):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, max_norm=None, norm_type=2):
        super(AdamWWithClip, self).__init__(
            params, lr, betas, eps, weight_decay, amsgrad)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def step(self, closure=None):

        if self.max_norm is not None:
            for group in self.param_groups:
                clip_grad_norm_(group['params'], self.max_norm, self.norm_type)
        super(AdamWWithClip, self).step(closure)

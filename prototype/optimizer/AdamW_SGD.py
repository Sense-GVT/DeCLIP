import torch
import math
from torch.optim.optimizer import Optimizer, required
from linklink.fp16 import FP16_Optimizer


class AdamW_SGD(Optimizer):
    r"""Implements layer-wise adaptive rate scaling for SGD, based on
    `"Large Batch Training of Convolutional Networks" <https://arxiv.org/abs/1708.03888>`_

    Arguments:
        - params (:obj:`iterable`): iterable of parameters to optimize or dicts defining parameter groups
        - lr (:obj:`float`): learning rate
        - momentum (:obj:`float`, optional): momentum factor (default: 0)
        - weight_decay (:obj:`float`, optional): weight decay (L2 penalty) (default: 0)
        - dampening (:obj:`float`, optional): dampening for momentum (default: 0)
        - eta(:obj:`float`): LARS coefficient (default 0.001)
        - nesterov (:obj:`bool`, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)

        super(AdamW_SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW_SGD, self).__setstate__(state)

    @torch.no_grad()
    def SGD_step(self, group):
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = p.grad
            if weight_decay != 0:
                d_p = d_p.add(p, alpha=weight_decay)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                if nesterov:
                    d_p = d_p.add(buf, alpha=momentum)
                else:
                    d_p = buf

            p.add_(d_p, alpha=-group['lr'])
        return

    @torch.no_grad()
    def AdamW_step(self, group):
        for p in group['params']:
            if p.grad is None:
                continue

            # Perform stepweight decay
            p.data.mul_(1 - group['lr'] * group['weight_decay'])

            # Perform optimization step
            grad = p.grad.data
            if grad.is_sparse:
                raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
            amsgrad = group['amsgrad']

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = max_exp_avg_sq.sqrt().add_(group['eps'])
            else:
                denom = exp_avg_sq.sqrt().add_(group['eps'])

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

            p.data.addcdiv_(-step_size, exp_avg, denom)
        return

    # @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            - closure (:obj:`callable`, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group['optimizer_type'] == 'SGD':
                self.SGD_step(group)
            elif group['optimizer_type'] == 'AdamW':
                self.AdamW_step(group)
            else:
                raise ValueError("Invalid optimizer type: {}".format(group['optimizer_type']))

        return loss



class FP16AdamW_SGD(FP16_Optimizer):
    def __init__(self, params, lr=required,
                 amsgrad=False, loss_scale='dynamic', verbose=False):

        optimizer = AdamW_SGD(params, lr=lr)
        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)
        super(FP16AdamW_SGD, self).__init__(optimizer,
                                            static_loss_scale=static_loss_scale,
                                            dynamic_loss_scale=dynamic_loss_scale,
                                            verbose=verbose)

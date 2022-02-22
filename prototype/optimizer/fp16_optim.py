from torch.optim.optimizer import required
from torch.optim import SGD, RMSprop, AdamW
from linklink.fp16 import FP16_Optimizer
# from .adam_clip import AdamWithClip, AdamWWithClip


class FP16SGD(FP16_Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum) in FP16.

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Arguments:
        - params (:obj:`iterable`): iterable of parameters to optimize or dicts defining parameter groups
        - lr (:obj:`float`): learning rate
        - momentum (:obj:`float`, optional): momentum factor (default: 0)
        - weight_decay (:obj:`float`, optional): weight decay (L2 penalty) (default: 0)
        - dampening (:obj:`float`, optional): dampening for momentum (default: 0)
        - nesterov (:obj:`bool`, optional): enables Nesterov momentum (default: False)
        - loss_scale (:obj:`sting`, default: :obj:`dynamic`): type to scale loss

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, loss_scale='dynamic', verbose=False):

        optimizer = SGD(params, lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)

        super(FP16SGD, self).__init__(optimizer,
                                      static_loss_scale=static_loss_scale,
                                      dynamic_loss_scale=dynamic_loss_scale,
                                      verbose=verbose)


class FP16RMSprop(FP16_Optimizer):
    r"""Implements RMSprop algorithm in FP16.

    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

    The implementation here takes the square root of the gradient average before
    adding epsilon (note that TensorFlow interchanges these two operations). The effective
    learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
    is the scheduled learning rate and :math:`v` is the weighted moving average
    of the squared gradient.

    Arguments:
        - params (:obj:`iterable`): iterable of parameters to optimize or dicts defining parameter groups
        - lr (:obj:`float`, optional): learning rate (default: 1e-2)
        - momentum (:obj:`float`, optional): momentum factor (default: 0)
        - alpha (:obj:`float`, optional): smoothing constant (default: 0.99)
        - eps (:obj:`float`, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        - centered (:obj:`bool`, optional) : if ``True``, compute the centered RMSProp, \
          the gradient is normalized by an estimation of its variance
        - weight_decay (:obj:`float`, optional): weight decay (L2 penalty) (default: 0)
        - loss_scale (:obj:`sting`, default: :obj:`dynamic`): type to scale loss

    """

    def __init__(self, params, lr=required, alpha=0.99, eps=1e-08,
                 weight_decay=0, momentum=0, centered=False, loss_scale='dynamic', verbose=False):

        optimizer = RMSprop(params, lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay,
                            momentum=momentum, centered=centered)

        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)

        super(FP16RMSprop, self).__init__(optimizer,
                                          static_loss_scale=static_loss_scale,
                                          dynamic_loss_scale=dynamic_loss_scale,
                                          verbose=verbose)


# class FP16Adam(FP16_Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
#                  amsgrad=False, max_norm=None, norm_type=2, loss_scale='dynamic', verbose=False):

#         optimizer = AdamWithClip(params, lr=lr, betas=betas, eps=eps,
#                                  weight_decay=weight_decay, amsgrad=amsgrad,
#                                  max_norm=max_norm, norm_type=norm_type)

#         dynamic_loss_scale = False
#         static_loss_scale = 1.0
#         if loss_scale == 'dynamic':
#             dynamic_loss_scale = True
#         else:
#             static_loss_scale = float(loss_scale)

#         super(FP16Adam, self).__init__(optimizer,
#                                        static_loss_scale=static_loss_scale,
#                                        dynamic_loss_scale=dynamic_loss_scale,
#                                        verbose=verbose)


# class FP16AdamW(FP16_Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
#                  amsgrad=False, max_norm=None, norm_type=2, loss_scale='dynamic', verbose=False):

#         optimizer = AdamWWithClip(params, lr=lr, betas=betas, eps=eps,
#                                   weight_decay=weight_decay, amsgrad=amsgrad,
#                                   max_norm=max_norm, norm_type=norm_type)

#         dynamic_loss_scale = False
#         static_loss_scale = 1.0
#         if loss_scale == 'dynamic':
#             dynamic_loss_scale = True
#         else:
#             static_loss_scale = float(loss_scale)

#         super(FP16AdamW, self).__init__(optimizer,
#                                         static_loss_scale=static_loss_scale,
#                                         dynamic_loss_scale=dynamic_loss_scale,
#                                         verbose=verbose)

class FP16AdamW(FP16_Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 amsgrad=False, loss_scale='dynamic', verbose=False):

        optimizer = AdamW(params, lr=lr, betas=betas, eps=eps,
                          weight_decay=weight_decay, amsgrad=amsgrad)
        dynamic_loss_scale = False
        static_loss_scale = 1.0
        if loss_scale == 'dynamic':
            dynamic_loss_scale = True
        else:
            static_loss_scale = float(loss_scale)
        super(FP16AdamW, self).__init__(optimizer,
                                        static_loss_scale=static_loss_scale,
                                        dynamic_loss_scale=dynamic_loss_scale,
                                        verbose=verbose)

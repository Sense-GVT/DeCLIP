import os
import copy
import logging
import torch
import linklink as link
from collections import defaultdict
import numpy as np
try:
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    print('Import metrics failed!')

from .dist import simple_group_split
import yaml
from easydict import EasyDict

_logger = None
_logger_fh = None
_logger_names = []


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reduce_update(self, tensor, num=1):
        link.allreduce(tensor)
        self.update(tensor.item(), num=num)

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val*num
            self.count += num
            self.avg = self.sum / self.count


def makedir(path):
    if link.get_rank() == 0 and not os.path.exists(path):
        os.makedirs(path)
    link.barrier()


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # config = yaml.safe_load(f)
    config = EasyDict(config)
    return config


class RankFilter(logging.Filter):
    def filter(self, record):
        return False


def create_logger(log_file, level=logging.INFO):
    global _logger, _logger_fh
    if _logger is None:
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        _logger.setLevel(level)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger_fh = fh
    else:
        _logger.removeHandler(_logger_fh)
        _logger.setLevel(level)

    return _logger


def get_logger(name, level=logging.INFO):
    global _logger_names
    logger = logging.getLogger(name)
    if name in _logger_names:
        return logger

    _logger_names.append(name)
    if link.get_rank() > 0:
        logger.addFilter(RankFilter())

    return logger


def get_bn(config):
    if config.use_sync_bn:
        group_size = config.kwargs.group_size
        var_mode = config.kwargs.var_mode
        if group_size == 1:
            bn_group = None
        else:
            world_size, rank = link.get_world_size(), link.get_rank()
            assert world_size % group_size == 0
            bn_group = simple_group_split(
                world_size, rank, world_size // group_size)

        del config.kwargs['group_size']
        config.kwargs.group = bn_group
        config.kwargs.var_mode = (
            link.syncbnVarMode_t.L1 if var_mode == 'L1' else link.syncbnVarMode_t.L2)

        def BNFunc(*args, **kwargs):
            return link.nn.SyncBatchNorm2d(*args, **kwargs, **config.kwargs)

        return BNFunc
    else:
        def BNFunc(*args, **kwargs):
            return torch.nn.BatchNorm2d(*args, **kwargs, **config.kwargs)
        return BNFunc


def get_norm_layer(config):
    if config.type == 'GroupNorm':
        def NormLayer(num_channels, *args, **kwargs):
            return torch.nn.GroupNorm(
                num_channels=num_channels, *args, **kwargs, **config.kwargs)
    else:
        assert False
    return NormLayer


def count_params(model):
    logger = get_logger(__name__)

    total = sum(p.numel() for p in model.parameters())
    conv = 0
    fc = 0
    others = 0
    for name, m in model.named_modules():
        # skip non-leaf modules
        if len(list(m.children())) > 0:
            continue
        num = sum(p.numel() for p in m.parameters())
        if isinstance(m, torch.nn.Conv2d):
            conv += num
        elif isinstance(m, torch.nn.Linear):
            fc += num
        else:
            others += num

    M = 1e6

    logger.info('total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'
                .format(total/M, conv/M, fc/M, others/M))


def count_flops(model, input_shape):
    try:
        from prototype.model.layer import CondConv2d
        from prototype.model.layer import WeightNet, WeightNet_DW
    except NotImplementedError:
        print('Check whether the file exists!')

    logger = get_logger(__name__)

    flops_dict = {}

    def make_conv2d_hook(name):

        def conv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                / m.stride[1] / m.stride[1] / m.groups
            flops_dict[name] = int(flops)

        return conv2d_hook

    def make_condconv2d_hook(name):

        def condconv2d_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            k, oc, c, kh, kw = m.weight.size()
            if m.combine_kernel:
                # flops of combining kernel
                flops_ck = n * oc * c * kh * kw * k
                # flops of group convolution: one group for each sample
                # input: n*c*h*w
                # weight: (n*oc)*c*kh*kw
                # groups: n
                flops_conv = n * h * w * oc * c * kh * kw / m.stride / m.stride / m.groups
                flops_dict[name] = int(flops_ck + flops_conv)
            else:
                # flops of group convolution: one group for each expert
                # input: n*(c*k)*h*w
                # weight: (c*k)*oc*kh*kw
                # groups: k
                flops_conv = k * n * h * w * c * oc * kh * kw / m.stride / m.stride / m.groups
                # flops of combining features
                flops_cf = n * h * w * oc * k
                flops_dict[name] = int(flops_conv + flops_cf)

        return condconv2d_hook

    def make_weightnet_hook(name):
        def weightnet_hook(m, input):
            n, _, h, w = input[0].size(0), input[0].size(
                1), input[0].size(2), input[0].size(3)
            oc = m.oup if hasattr(m, 'oup') else m.inp
            group = 1 if hasattr(m, 'oup') else m.inp
            # flops of group convolution: one group for each sample
            # input: n*c*h*w
            # weight: (n*oc)*c*kh*kw
            flops_dict[name] = n * h * w * oc * m.inp * m.ksize * m.ksize / m.stride / m.stride / group

        return weightnet_hook

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            h = m.register_forward_pre_hook(make_conv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, CondConv2d):
            h = m.register_forward_pre_hook(make_condconv2d_hook(name))
            hooks.append(h)
        elif isinstance(m, WeightNet) or isinstance(m, WeightNet_DW):
            h = m.register_forward_pre_hook(make_weightnet_hook(name))
            hooks.append(h)

    input = torch.zeros(*input_shape).cuda()

    model.eval()
    with torch.no_grad():
        _ = model(input)

    model.train()
    total_flops = 0
    for k, v in flops_dict.items():
        # logger.info('module {}: {}'.format(k, v))
        total_flops += v
    logger.info('total FLOPS: {:.2f}M'.format(total_flops/1e6))

    for h in hooks:
        h.remove()

def param_name(name, suffix):
    return name + '.' + suffix if len(name)>0 else suffix

def param_group_all(model, config, default_config={}):
    logger = get_logger(__name__)
    pgroup_normal = []
    pgroup = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': [], 'ln_w': [], 'ln_b': []}
    names = {'bn_w': [], 'bn_b': [], 'conv_b': [], 'linear_b': [], 'ln_w': [], 'ln_b': []}
    if 'conv_dw_w' in config:
        pgroup['conv_dw_w'] = []
        names['conv_dw_w'] = []
    if 'conv_dw_b' in config:
        pgroup['conv_dw_b'] = []
        names['conv_dw_b'] = []
    if 'conv_dense_w' in config:
        pgroup['conv_dense_w'] = []
        names['conv_dense_w'] = []
    if 'conv_dense_b' in config:
        pgroup['conv_dense_b'] = []
        names['conv_dense_b'] = []
    if 'linear_w' in config:
        pgroup['linear_w'] = []
        names['linear_w'] = []
    if 'logit_scale' in config:
        pgroup['logit_scale'] = []
        names['logit_scale'] = []
    if 'bias' in config:  # each x.bias
        pgroup['bias'] = []
        names['bias'] = []

    names_all = []
    type2num = defaultdict(lambda: 0)
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            if m.bias is not None:
                if 'bias' in pgroup:
                    pgroup['bias'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['bias'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
                elif 'conv_dw_b' in pgroup and m.groups == m.in_channels:
                    pgroup['conv_dw_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['conv_dw_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias(dw)'] += 1
                elif 'conv_dense_b' in pgroup and m.groups == 1:
                    pgroup['conv_dense_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['conv_dense_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias(dense)'] += 1
                else:
                    pgroup['conv_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['conv_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
            if 'conv_dw_w' in pgroup and m.groups == m.in_channels:
                pgroup['conv_dw_w'].append(m.weight)
                names_all.append(param_name(name, 'weight'))
                names['conv_dw_w'].append(param_name(name, 'weight'))
                type2num[m.__class__.__name__+'.weight(dw)'] += 1
            elif 'conv_dense_w' in pgroup and m.groups == 1:
                pgroup['conv_dense_w'].append(m.weight)
                names_all.append(param_name(name, 'weight'))
                names['conv_dense_w'].append(param_name(name, 'weight'))
                type2num[m.__class__.__name__+'.weight(dense)'] += 1
        elif isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                if 'bias' in pgroup:
                    pgroup['bias'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['bias'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
                else:
                    pgroup['linear_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['linear_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
            if 'linear_w' in pgroup:
                pgroup['linear_w'].append(m.weight)
                names_all.append(param_name(name, 'weight'))
                names['linear_w'].append(param_name(name, 'weight'))
                type2num[m.__class__.__name__+'.weight'] += 1
        elif (isinstance(m, torch.nn.BatchNorm2d)
              or isinstance(m, torch.nn.BatchNorm1d)
              or isinstance(m, link.nn.SyncBatchNorm2d)):
            if m.weight is not None:
                pgroup['bn_w'].append(m.weight)
                names_all.append(param_name(name, 'weight'))
                names['bn_w'].append(param_name(name, 'weight'))
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                if 'bias' in pgroup:
                    pgroup['bias'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['bias'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
                else:
                    pgroup['bn_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['bn_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
        elif (isinstance(m, torch.nn.LayerNorm)):
            if m.weight is not None:
                pgroup['ln_w'].append(m.weight)
                names_all.append(param_name(name, 'weight'))
                names['ln_w'].append(param_name(name, 'weight'))
                type2num[m.__class__.__name__+'.weight'] += 1
            if m.bias is not None:
                if 'bias' in pgroup:
                    pgroup['bias'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['bias'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
                else:
                    pgroup['ln_b'].append(m.bias)
                    names_all.append(param_name(name, 'bias'))
                    names['ln_b'].append(param_name(name, 'bias'))
                    type2num[m.__class__.__name__+'.bias'] += 1
    for name, p in model.named_parameters():
        if link.get_rank() == 0:
            print('solve', name, flush=True)
        if 'logit_scale' in config and 'logit_scale' in name:
            pgroup['logit_scale'].append(p)
            names_all.append(name)
            names['logit_scale'].append(name)
        if name not in names_all:
            pgroup_normal.append(p)

    param_groups = [{'params': pgroup_normal, **default_config}]
    for ptype in pgroup.keys():
        if ptype in config.keys():
            special_config = copy.deepcopy(default_config)
            special_config.update(config[ptype])
            param_groups.append({'params': pgroup[ptype], **special_config})
        else:
            param_groups.append({'params': pgroup[ptype], **default_config})

        logger.info(ptype)
        for k, v in param_groups[-1].items():
            if k == 'params':
                logger.info('   params: {}'.format(len(v)))
            else:
                logger.info('   {}: {}'.format(k, v))

    for ptype, pconf in config.items():
        logger.info('names for {}({}): {}'.format(
            ptype, len(names[ptype]), names[ptype]))

    return param_groups, type2num


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def detailed_metrics(output, target):
    precision_class = precision_score(target, output, average=None)
    recall_class = recall_score(target, output, average=None)
    f1_class = f1_score(target, output, average=None)
    precision_avg = precision_score(target, output, average='micro')
    recall_avg = recall_score(target, output, average='micro')
    f1_avg = f1_score(target, output, average='micro')
    return precision_class, recall_class, f1_class, precision_avg, recall_avg, f1_avg


def load_state_model(model, state):

    logger = get_logger(__name__)
    logger.info('======= loading model state... =======')

    load_result = model.load_state_dict(state, strict=False)

    state_keys = set(state.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = model_keys - state_keys
    for k in missing_keys:
        logger.warn(f'missing key: {k}')

    # logger.info(load_result)



def load_state_optimizer(optimizer, state):

    logger = get_logger(__name__)
    logger.info('======= loading optimizer state... =======')

    optimizer.load_state_dict(state)


def modify_state(state, config):
    if hasattr(config, 'key'):
        for key in config['key']:
            if key == 'optimizer':
                state.pop(key)
            elif key == 'last_iter':
                state['last_iter'] = 0
            elif key == 'ema':
                state.pop('ema')

    if hasattr(config, 'model'):
        for module in config['model']:
            state['model'].pop(module)
    return state


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(input, target, alpha=0.0):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(input.size()[0]).cuda()

    target_a = target
    target_b = target[rand_index]

    # generate mixed sample
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
    return input, target_a, target_b, lam


import logging
import math
import collections
import copy
import torch

logger = logging.getLogger(__name__)


def _get_conv_2d_output_shape(conv_args, x):
    # When input is empty, we want to return a empty tensor with "correct" shape,
    # So that the following operations will not panic
    # if they check for the shape of the tensor.
    # This computes the height and width of the output tensor
    output_shape = [
        (i + 2 * p - (di * (k - 1) + 1)) // s + 1
        for i, p, di, k, s in zip(
            x.shape[-2:],
            conv_args.padding,
            conv_args.dilation,
            conv_args.kernel_size,
            conv_args.stride,
        )
    ]
    output_shape = [x.shape[0], conv_args.out_channels] + output_shape
    return output_shape


def filter_kwargs(func, kwargs, log_skipped=True):
    """ Filter kwargs based on signature of `func`
        Return arguments that matches `func`
    """
    import inspect

    sig = inspect.signature(func)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]

    if log_skipped:
        skipped_args = [x for x in kwargs.keys() if x not in filter_keys]
        if skipped_args:
            logger.warning(
                f"Arguments {skipped_args} skipped for op {func.__name__}"
            )

    filtered_dict = {
        filter_key: kwargs[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs
    }
    return filtered_dict


def merge_unify_args(*args):
    from collections import ChainMap

    unified_args = [unify_args(x) for x in args]
    ret = dict(ChainMap(*unified_args))
    return ret


def unify_args(aargs):
    """ Return a dict of args """
    if aargs is None:
        return {}
    if isinstance(aargs, str):
        return {"name": aargs}
    assert isinstance(aargs, dict), f"args {aargs} must be a dict or a str"
    return aargs


def drop_connect_batch(inputs, drop_prob, training):
    """ Randomly drop batch during training """
    assert drop_prob < 1.0, f"Invalid drop_prob {drop_prob}"
    if not training or drop_prob == 0.0:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - drop_prob
    random_tensor = (
            torch.rand(
                [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
            )
            + keep_prob
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    # output = inputs * binary_tensor
    return output


def get_divisible_by(num, divisible_by, min_val=None):
    ret = int(num)
    if min_val is None:
        min_val = divisible_by
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((py2_round(num / divisible_by) or 1) * divisible_by)
        if ret < 0.9 * num:
            ret += divisible_by
    return ret


def update_dict(dest, src):
    """ Update the dict 'dest' recursively.
        Elements in src could be a callable function with signature
            f(key, curr_dest_val)
    """
    for key, val in src.items():
        if isinstance(val, collections.Mapping):
            # dest[key] could be None in the case of a dict
            cur_dest = dest.get(key, {}) or {}
            assert isinstance(cur_dest, dict), cur_dest
            dest[key] = update_dict(cur_dest, val)
        else:
            if callable(val) and key in dest:
                dest[key] = val(key, dest[key])
            else:
                dest[key] = val
    return dest


def merge(kwargs, **all_args):
    """ kwargs will override other arguments """
    return update_dict(all_args, kwargs)


def py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def get_merged_dict(base, *new_dicts):
    ret = copy.deepcopy(base)
    for x in new_dicts:
        assert isinstance(x, dict)
        update_dict(ret, x)
    return ret


def add_dropout(dropout_ratio):
    if dropout_ratio > 0:
        return torch.nn.Dropout(dropout_ratio)
    return None

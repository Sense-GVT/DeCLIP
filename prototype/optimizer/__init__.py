import linklink as link

from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam, AdamW  # noqa F401
from .lars import LARS  # noqa F401
from .AdamW_SGD import AdamW_SGD, FP16AdamW_SGD  # noqa F401
from .fp16_optim import FP16SGD, FP16RMSprop, FP16AdamW  # noqa F401
# from .adamw_sgd import AdamW_SGD
try:
    from linklink.optim import FusedFP16SGD
    from linklink.optim import FusedFP16AdamW
except ModuleNotFoundError:
#     print('import FusedFP16SGD failed, linklink version should >= 0.1.6')
    print('import FusedFP16SGD failed, FusedFP16AdamW replace')
    FusedFP16SGD = SGD
    FusedFP16AdamW = AdamW


def optim_entry(config):
    rank = link.get_rank()
#     if (config['type'] == 'FusedFP16SGD' and FusedFP16SGD is None) or \
#        (config['type'] == 'FusedFP16AdamW' and FusedFP16AdamW is None):
#         raise RuntimeError(
#             'FusedFP16SGD and FusedFP16AdamW are disabled due to linklink version, try using other optimizers')
#     if config['type'] == 'FusedFP16SGD' or config['type'] == 'FusedFP16AdamW' and rank > 0:
#         config['kwargs']['verbose'] = False
    return globals()[config['type']](**config['kwargs'])


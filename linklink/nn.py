import torch


# SyncBatchNorm2d = torch.nn.SyncBatchNorm
SyncBatchNorm2d = torch.nn.BatchNorm1d


class syncbnVarMode_t(object):
    L1 = None
    L2 = None
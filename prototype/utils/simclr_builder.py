import torch
import torch.nn as nn
import linklink as link


class SimCLR(nn.Module):
    def __init__(self, encoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        dim_fc = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(dim_fc, dim_fc), nn.ReLU(), self.encoder.fc)

    def forward(self, x):
        im_i, im_j = torch.split(x, [3, 3], dim=1)
        im_i = im_i.contiguous()
        im_j = im_j.contiguous()

        logits_i = self.encoder(im_i)
        logits_j = self.encoder(im_j)

        # TOFIX: allgather does not support backward
        # logits_i = concat_all_gather(logits_i)
        # logits_j = concat_all_gather(logits_j)

        return logits_i, logits_j


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(link.get_world_size())]
    link.allgather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output

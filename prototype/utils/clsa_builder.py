import torch
import torch.nn as nn
import linklink as link
from .dist import simple_group_split

class CLSA(nn.Module):
    def __init__(self, encoder_q, encoder_k, K=65536, m=0.999, T=0.07, mlp=False, group_size=8, ratio=1.0, avg=False):
        """
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        group_size: size of the group to use ShuffleBN (default: 8, shuffle data across all gpus)
        """
        super(CLSA, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.ratio = ratio
        self.avg = avg
        dim = encoder_q.num_classes

        self.encoder_q = encoder_q
        self.encoder_k = encoder_k

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = nn.CrossEntropyLoss()

        rank = link.get_rank()
        world_size = link.get_world_size()
        self.group_size = world_size if group_size is None else min(
            world_size, group_size)

        assert world_size % self.group_size == 0
        self.group_idx = simple_group_split(
            world_size, rank, world_size // self.group_size)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys, self.group_size, self.group_idx)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, self.group_size, self.group_idx)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        link.broadcast(idx_shuffle, 0, group_idx=self.group_idx)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = link.get_rank() % self.group_size
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x, self.group_size, self.group_idx)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = link.get_rank() % self.group_size
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, img_stronger_aug_list):
        # im_q, im_k = torch.split(input, [3, 3], dim=1)
        # im_q = im_q.contiguous()
        # im_k = im_k.contiguous()

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss_contrastive = self.criterion(logits, labels)

        # compute ddm loss below
        # get P(Zk, Zi')
        loss_ddm = 0

        if type(self.avg) is bool:
            p_weak = nn.functional.softmax(logits, dim=-1)
            for img_s in img_stronger_aug_list:

                # compute query features
                q_s = self.encoder_q(img_s)
                q_s = nn.functional.normalize(q_s, dim=1)

                # compute logits using the same set of code above
                l_pos_stronger_aug = torch.einsum('nc,nc->n', [q_s, k]).unsqueeze(-1)
                # negative logits: NxK
                l_neg_stronger_aug = torch.einsum('nc,ck->nk', [q_s, self.queue.clone().detach()])

                # logits: Nx(1+K)
                logits_s = torch.cat([l_pos_stronger_aug, l_neg_stronger_aug], dim=1)
                logits_s /= self.T

                # compute nll loss below as -P(q, k) * log(P(q_s, k))
                log_p_s = nn.functional.log_softmax(logits_s, dim=-1)

                nll = -1.0 * torch.einsum('nk,nk->n', [p_weak, log_p_s])
                loss_ddm = loss_ddm + torch.mean(nll) # average over the batch dimension

        loss = loss_contrastive + self.ratio * loss_ddm

        if self.avg == True: # avg over contrastive num
            loss /= (len(img_stronger_aug_list) + 1)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return loss

# utils
@torch.no_grad()
def concat_all_gather(tensor, group_size, group_idx):
    """gather the given tensor across the group"""

    tensors_gather = [torch.ones_like(tensor) for _ in range(group_size)]
    link.allgather(tensors_gather, tensor, group_idx)

    output = torch.cat(tensors_gather, dim=0)
    return output

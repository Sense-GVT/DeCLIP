import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import linklink as link

class NT_Xent(_Loss):
    r"""The normalized temperature-scaled cross entropy loss, based on
    `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
    """

    def __init__(self, batch_size, temperature=0.5):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = self.mask_correlated_samples(batch_size)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        p1 = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(p1.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(
            self.batch_size * 2, 1
        )
        negative_samples = sim[self.mask].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        return loss


class NT_Xent_gather(_Loss):
    r"""The normalized temperature-scaled cross entropy loss, based on
    `"A Simple Framework for Contrastive Learning of Visual Representations" <https://arxiv.org/abs/2002.05709>`_
    """

    def __init__(self, batch_size, temperature=0.1):
        super(NT_Xent_gather, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask_positive = None
        self.mask_negative = None

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_ib, z_j, z_jb, temperature=None):  # z_ib, z_jb: in-batch
        bs = z_i.shape[0]
        assert bs == self.batch_size
        l_bs = z_ib.shape[0]

        if temperature is None:
            temperature = self.temperature

        p0 = torch.cat((z_i, z_j), dim=0)
        p1 = torch.cat((z_ib, z_jb), dim=0)
        sim = self.similarity_f(p0.unsqueeze(1), p1.unsqueeze(0)) / self.temperature

        if self.mask_positive is None:
            ids = torch.arange(0, bs, dtype=torch.long).to(z_i.device)
            labels = link.get_rank() * bs + torch.arange(0, bs, dtype=torch.long).to(z_i.device)
            # positive samples
            self.mask_positive = torch.zeros([bs*2, l_bs*2]).bool()
            self.mask_positive[ids+bs, labels] = 1
            self.mask_positive[ids, labels+l_bs] = 1
            # negative samples
            self.mask_negative = torch.ones([bs*2, l_bs*2]).bool()
            self.mask_negative[ids, labels] = 0
            self.mask_negative[ids+bs, labels] = 0
            self.mask_negative[ids, labels+l_bs] = 0
            self.mask_negative[ids+bs, labels+l_bs] = 0

        positive_samples = sim[self.mask_positive].reshape(self.batch_size * 2, -1)
        negative_samples = sim[self.mask_negative].reshape(self.batch_size * 2, -1)

        labels = torch.zeros(self.batch_size * 2).to(z_i.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= 2 * self.batch_size
        # import ipdb
        # ipdb.set_trace()
        return loss


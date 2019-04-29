import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


torch.manual_seed(0)
np.random.seed(0)

'''
this class will play more efficient negative sampling than take all samples in the batch as negative
for each sample, it only consider up to n negative samples in the batch, these samplings are more similar than others in the batch
with probability=p, it do efficient sampling while with probability=(1-p), it randomly select negatvie samples
'''
class NSHingeLoss(nn.Module):
    def __init__(self, p=1, negative_num=2, margin=1.0, reduction="mean"):
        super(NSHingeLoss, self).__init__()
        self.p = p
        self.margin=margin
        self.reduction=reduction
        self.negative_num = negative_num

    def forward(self, M):
        '''
        :param M: similarity matrix: [batch_size, batch_size]
        :return: the loss
        '''
        assert M.shape[0] == M.shape[1]
        batch_size = M.shape[0]
        negative_num = min(self.negative_num, batch_size - 1)

        predict = torch.diag(M).unsqueeze(-1)
        ns_prob = np.random.uniform(0, 1, 1)[0]
        if ns_prob >= self.p: # random sample
            negative_idx = torch.zeros((batch_size, negative_num))
            # get random sampling
            for i in range(batch_size):
                prob = np.ones((batch_size)) / (batch_size - 1)
                prob[i] = 0
                negative_idx[i] = torch.LongTensor(np.random.choice(batch_size, negative_num, p=prob))
            negative_idx = negative_idx.long()
        else: # max sample
            masked_M = M.masked_fill(torch.eye(batch_size, batch_size) == 1, -1e-9)
            _, negative_idx = torch.topk(masked_M, k=negative_num, dim=-1)

        negative_sample = torch.gather(M, 1, negative_idx)

        loss = self.margin + negative_sample - predict
        loss[loss < 0] = 0
        if self.reduction == "mean":
            loss = torch.sum(loss) / batch_size
        elif self.reduction == "sum":
            loss = torch.sum(loss)

        return loss

class MultiMarginLoss(nn.Module):
    def __init__(self, device, margin=1, reduction="mean"):
        super(MultiMarginLoss, self).__init__()
        self.criterion = nn.MultiMarginLoss(margin=margin, reduction=reduction)
        self.device = device
    def forward(self, M):
        # assert M.shape[0] == M.shape[1]
        batch_size = M.shape[0]
        # [1, batch_size]
        label = torch.from_numpy(np.array([0 for _ in range(batch_size)])).long().to(self.device)
        return self.criterion(M, label)


class CrossEntropyLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.device = device

    def forward(self, M):
        # assert M.shape[0] == M.shape[1]
        batch_size = M.shape[0]
        # [1, batch_size]
        label = torch.from_numpy(np.array([0 for _ in range(batch_size)])).long().to(self.device)
        return self.criterion(M, label)


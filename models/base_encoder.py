import sys
sys.path.append("/home/shuyanzh/workshop/cmu_lorelei_edl/")
import functools
import torch
from torch import nn
from utils.constant import DEVICE
from data_loader.data_loader import BaseBatch
from torch import optim

print = functools.partial(print, flush=True)
device = DEVICE

def create_optimizer(trainer, lr, model, lr_decay=False):
    if trainer == "adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif trainer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr)
    elif trainer == "sgd_mo":
        optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    elif trainer == "rmsp":
        optimizer = optim.RMSprop(model.parameters(), lr)
    else:
        raise NotImplementedError
    # if lr_decay:
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50, 100], gamma=0.3)
    # else:
    #     scheduler = None
    scheduler = None
    return optimizer, scheduler

class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        # self.src_trg_bl = nn.Parameter(torch.eye(self.hidden_size), requires_grad=True)
        # self.src_mid_bl = nn.Parameter(torch.eye(self.hidden_size), requires_grad=True)
        self.src_trg_bl = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
        self.src_mid_bl = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
        torch.nn.init.xavier_uniform_(self.src_trg_bl, gain=1)
        torch.nn.init.xavier_uniform_(self.src_mid_bl, gain=1)
        self.src_affine = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
        torch.nn.init.xavier_uniform_(self.src_affine, gain=1)
        self.trg_affine = nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        torch.nn.init.xavier_uniform_(self.trg_affine, gain=1)

    def calc_batch_similarity(self, batch: BaseBatch, trg_encoding_num, mid_encoding_num, proportion,
                              use_negative=False, use_mid=False):
        # [batch_size, hidden_state]
        src_encoded = self.calc_encode(batch, is_src=True)
        trg_encoded = self.calc_encode(batch, is_src=False)

        if batch.mega_flag:
            mega_encoded = self.calc_encode(batch, is_src=False, is_mega=True)
            # [batch_size * 2, hidden state ]
            trg_encoded = torch.cat((trg_encoded, mega_encoded), dim=0)

        # because this function is also called when calculate mega batch similarity, there is no need to use negative sample
        if use_negative:
            ns = batch.negative_num
        else:
            ns = None
        # if negative_sample is not none, it will move the correct answer to idx 0
        similarity = self.similarity_measure(src_encoded, trg_encoded, is_src_trg=True,
                                             split=False, pieces=0, negative_sample=ns, encoding_num=trg_encoding_num)
        # calc middle representation
        diff = None
        if use_mid and batch.mid_flag:
            p = proportion
            mid_encoded = self.calc_encode(batch, is_src=False, is_mid=True)
            similarity_src_mid = self.similarity_measure(src_encoded, mid_encoded,
                                                         is_src_trg=False, split=False, pieces=0, negative_sample=ns,
                                                         encoding_num=mid_encoding_num)
            cur_batch_size = similarity.shape[0]
            similarity[:, 1:int(cur_batch_size * p)] = \
                similarity_src_mid[:, 1:int(cur_batch_size * p)]
            similarity[:int(cur_batch_size * p), 0] = \
                similarity_src_mid[:int(cur_batch_size * p), 0]

        return similarity, diff

    def calc_encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def set_similarity_matrix(self):
        self.similarity_measure.set_src_trg_bl(self.src_trg_bl)
        self.similarity_measure.set_src_mid_bl(self.src_mid_bl)
        self.similarity_measure.set_src_affine(self.src_affine)
        self.similarity_measure.set_trg_affine(self.trg_affine)
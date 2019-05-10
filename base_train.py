import sys
sys.path.append("/home/shuyanzh/workshop/cmu_lorelei_edl/")
from collections import defaultdict
import functools
import torch
from torch import nn
from torch import optim
import random
import time
import numpy as np
import pickle
from similarity_calculator import Similarity
import argparse
from typing import List, Generator
from criterion import NSHingeLoss, MultiMarginLoss, CrossEntropyLoss
from collections import defaultdict
from itertools import combinations


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

print = functools.partial(print, flush=True)
PATIENT = 50
EPOCH_CHECK = 2
PP_VEC_SIZE = 22
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseBatch:
    def __init__(self):
        self.src_flag=False
        self.trg_flag=False
        self.mega_flag=False
        self.mid_flag=False
        self.negative_num=0
        self.src_gold_kb_ids = None
        self.trg_kb_ids = None

    def set_src(self, *args, **kwargs):
        pass
    def set_trg(self, *args, **kwargs):
        pass
    def set_mega(self, *args, **kwargs):
        pass
    def set_mid(self, *args, **kwargs):
        pass
    def get_all(self, *args, **kwargs):
        pass
    def get_src(self, *args, **kwargs):
        pass
    def get_trg(self, *args, **kwargs):
        pass
    def get_mega(self, *args, **kwargs):
        pass
    def get_mid(self, *args, **kwargs):
        pass
    def to(self,  *args, **kwargs):
        pass


class FileInfo:
    def __init__(self):
        self.src_file_name = None
        self.src_str_idx = None
        self.src_id_idx = None
        self.trg_file_name = None
        self.trg_str_idx = None
        self.trg_id_idx = None
        self.mid_file_name = None
        self.mid_str_idx = None
        self.mid_id_idx = None

    def set_all(self, file_name, src_str_idx, trg_str_idx, id_idx, type_idx):
        self.src_file_name = file_name
        self.trg_file_name = file_name
        self.src_str_idx = int(src_str_idx)
        self.trg_str_idx = int(trg_str_idx)
        self.src_id_idx = int(id_idx)
        self.trg_id_idx = int(id_idx)
        self.trg_type_idx = int(type_idx)

    def set_src(self, file_name, str_idx, id_idx):
        self.src_file_name = file_name
        self.src_str_idx = int(str_idx)
        self.src_id_idx = int(id_idx)

    def set_trg(self, file_name, str_idx, id_idx, type_idx):
        self.trg_file_name = file_name
        self.trg_str_idx = int(str_idx)
        self.trg_id_idx = int(id_idx)
        self.trg_type_idx = int(type_idx)

    def set_mid(self, file_name, str_idx, id_idx, type_idx):
        self.mid_file_name = file_name
        self.mid_str_idx = int(str_idx)
        self.mid_id_idx = int(id_idx)
        self.mid_type_idx = int(type_idx)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def calc_batch_similarity(self, batch:BaseBatch, trg_encoding_num, mid_encoding_num, proportion, use_negative=False, use_mid=False):
        #[batch_size, hidden_state]
        src_encoded = self.calc_encode(batch, is_src=True)
        trg_encoded = self.calc_encode(batch, is_src=False)

        if batch.mega_flag:
            mega_encoded = self.calc_encode(batch, is_src=False, is_mega=True)
            #[batch_size * 2, hidden state ]
            trg_encoded = torch.cat((trg_encoded, mega_encoded), dim=0)
            
        # because this function is also called when calculate mega batch similarity, there is no need to use negative sample
        if use_negative:
            ns = batch.negative_num
        else:
            ns = None
        # if negative_sample is not none, it will move the correct answer to idx 0
        similarity = self.similarity_measure(src_encoded, trg_encoded, self.bilinear, split=False, pieces=0, negative_sample=ns, encoding_num=trg_encoding_num)
        # calc middle representation
        diff = None
        if use_mid and batch.mid_flag:
            p = proportion
            mid_encoded = self.calc_encode(batch, is_src=False, is_mega=False, is_mid=True)
            similarity_src_mid = self.similarity_measure(src_encoded, mid_encoded, self.bilinear_mid, split=False, pieces=0, negative_sample=ns, encoding_num=mid_encoding_num)
            cur_batch_size =similarity.shape[0]
            similarity[:, 1:int(cur_batch_size * p)] =\
                similarity_src_mid[:, 1:int(cur_batch_size * p)]
            similarity[:int(cur_batch_size * p), 0] = \
                similarity_src_mid[:int(cur_batch_size * p), 0]

        return similarity, diff

    def calc_encode(self, *args, **kwargs)->torch.Tensor:
        pass


class BaseDataLoader:
        def __init__(self, is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab, pad_str,
                     train_file:FileInfo, dev_file:FileInfo, test_file:FileInfo, trg_encoding_num, mid_encoding_num):
            self.batch_size = batch_size
            self.train_file = train_file
            self.dev_file = dev_file
            self.use_panphon = use_panphon
            self.map_file = map_file
            self.pad_str = pad_str
            self.pad_idx = 0
            self.train_file = train_file
            self.dev_file = dev_file
            self.test_file = test_file
            self.mega_batch_size = mega_size * batch_size
            self.use_mid = use_mid
            self.share_vocab = share_vocab
            self.trg_encoding_num = trg_encoding_num
            self.mid_encoding_num = mid_encoding_num
            if is_train:
                self.init_train()
            else:
                self.init_test()

        def init_train(self):
            self.x2i_src = defaultdict(lambda: len(self.x2i_src))
            self.x2i_trg = defaultdict(lambda: len(self.x2i_trg))
            # make sure pad is 0
            self.x2i_src[self.pad_str]
            self.x2i_trg[self.pad_str]
            self.train_src = list(self.load_data(self.train_file.src_file_name, self.train_file.src_str_idx, self.train_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
            self.train_trg = list(self.load_data(self.train_file.trg_file_name, self.train_file.trg_str_idx, self.train_file.trg_id_idx, is_src=False, encoding_num=self.trg_encoding_num, type_idx=self.train_file.trg_type_idx))
            # save map
            self.save_map(self.x2i_src, self.map_file + "_src.pkl")
            self.save_map(self.x2i_trg, self.map_file + "_trg.pkl")

            if self.use_mid:
                if self.share_vocab:
                    self.x2i_mid = self.x2i_src
                else:
                    self.x2i_mid = defaultdict(lambda: len(self.x2i_mid))
                    self.x2i_mid[self.pad_str]
                self.train_mid = list(self.load_data(self.train_file.mid_file_name, self.train_file.mid_str_idx, self.train_file.mid_id_idx,
                                                     is_src=False, is_mid=True, encoding_num=self.mid_encoding_num, type_idx=self.train_file.mid_type_idx))
                self.save_map(self.x2i_mid, self.map_file + "_mid.pkl")
                self.mid_vocab_size = len(self.x2i_mid)
                self.x2i_mid = defaultdict(lambda: self.x2i_mid[self.pad_str], self.x2i_mid)
            else:
                self.train_mid = None
                self.mid_vocab_size = 0

            self.non_neg_mask = self.get_non_negative_mask()

            # sort training data by input length
            self.src_vocab_size = len(self.x2i_src)
            self.trg_vocab_size = len(self.x2i_trg)
            self.x2i_src = defaultdict(lambda: self.x2i_src[self.pad_str], self.x2i_src)
            self.x2i_trg = defaultdict(lambda: self.x2i_trg[self.pad_str], self.x2i_trg)

            if self.dev_file:
                self.dev_src = list(self.load_data(self.dev_file.src_file_name, self.dev_file.src_str_idx, self.dev_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
                self.dev_trg = list(self.load_data(self.dev_file.trg_file_name, self.dev_file.trg_str_idx, self.dev_file.trg_id_idx, is_src=False, encoding_num=self.trg_encoding_num, type_idx=self.dev_file.trg_type_idx))
                n = min(len(self.dev_src), 2000)
                self.dev_src, self.dev_trg = self.dev_src[:n], self.dev_trg[:n]
                if self.use_mid:
                    self.dev_mid = list(self.load_data(self.dev_file.mid_file_name, self.dev_file.mid_str_idx,
                                                       self.dev_file.mid_id_idx, is_src=False, is_mid=True, encoding_num=self.mid_encoding_num, type_idx=self.dev_file.mid_type_idx))
                    self.dev_mid = self.dev_mid[:n]
                else:
                    self.dev_mid = None
            else:
                self.dev_src, self.dev_trg, self.dev_mid = None, None, None

        def get_non_negative_mask(self):
            id_idx_map = defaultdict(list)
            for idx, (_, kb_id) in enumerate(self.train_src):
                id_idx_map[kb_id].append(idx)
            mask = torch.zeros((len(self.train_src), len(self.train_src)))
            for _, idx_list in id_idx_map.items():
                idx_pairs = combinations(idx_list, 2)
                for i, j in idx_pairs:
                    mask[i,j] = 1
                    mask[j,i] = 1
            mask += torch.eye(len(self.train_src), len(self.train_src))
            mask = mask.long()
            # mask = mask.to(device)
            return mask

        def init_test(self):
            self.x2i_src = self.load_map(self.map_file + "_src.pkl")
            self.x2i_trg = self.load_map(self.map_file + "_trg.pkl")
            if self.use_mid:
                self.x2i_mid = self.load_map(self.map_file + "_mid.pkl")
            else:
                self.x2i_mid = None
            self.i2c_src = {v: k for k, v in self.x2i_src.items()}
            self.i2c_trg = {v: k for k, v in self.x2i_trg.items()}
            if self.test_file.src_file_name is not None:
                self.test_src = list(self.load_data(self.test_file.src_file_name, self.test_file.src_str_idx, self.test_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
            if self.test_file.trg_file_name is not None:
                self.test_trg = list(self.load_data(self.test_file.trg_file_name, self.test_file.trg_str_idx, self.test_file.trg_id_idx, is_src=False, encoding_num=self.trg_encoding_num, type_idx=self.test_file.trg_type_idx))
            if self.test_file.mid_file_name is not None:
                self.test_mid = list(self.load_data(self.test_file.mid_file_name, self.test_file.mid_str_idx, self.test_file.mid_id_idx, is_src=False, is_mid=True, encoding_num=self.mid_encoding_num, type_idx=self.test_file.mid_type_idx))

        def load_all_data(self, file_name, str_idx, id_idx, x2i_map, encoding_num, type_idx):
            pass

        def load_data(self, file_name, str_idx, id_idx, is_src, encoding_num, type_idx, is_mid=False):
            if is_src:
                x2i_map = self.x2i_src
            else:
                x2i_map = self.x2i_trg
            if is_mid:
                x2i_map = self.x2i_mid
            return self.load_all_data(file_name, str_idx, id_idx, x2i_map, encoding_num, type_idx)

        def transform_one_batch(self, *args, **kwargs) -> list:
            pass

        def new_batch(self) -> BaseBatch:
            pass

        # data from one side
        def prepare_batch(self, side_data, data_idx, encoding_num):
            # this is a list of words
            words = [side_data[idx][0] for idx in data_idx]
            # expand words to list
            all_words = []
            for i in range(encoding_num):
                all_words += [words[idx][i] for idx in range(len(words))]
            assert len(all_words) == len(words) * encoding_num


            kb_ids = [side_data[idx][1] for idx in data_idx]
            batch_info = self.transform_one_batch(all_words)

            return batch_info, kb_ids

        def create_batch(self, dataset, data_src=None, data_trg=None, data_mega=None, data_mid=None) -> List[BaseBatch]:
            batches = []
            non_none = [x for x in [data_src, data_trg, data_mid] if x is not None][0]
            data_idx = [i for i in range(len(non_none))]
            if dataset == "train":
                random.shuffle(data_idx)
            for i in range(0, len(data_idx), self.batch_size):
                batch = self.new_batch()
                cur_size = min(self.batch_size, len(data_idx) - i)
                cur_data_idx = data_idx[i:i + cur_size]
                if data_src is not None:
                    batch_info, src_gold_kb_ids = self.prepare_batch(data_src, cur_data_idx, encoding_num=1)
                    batch.set_src(*batch_info, src_gold_kb_ids)
                if data_trg is not None:
                    batch_info, trg_kb_ids = self.prepare_batch(data_trg, cur_data_idx, encoding_num=self.trg_encoding_num)
                    batch.set_trg(*batch_info, trg_kb_ids)
                if data_mega is not None:
                    #TODO this needs to be fix with multiple encodings!
                    batch_info, trg_kb_ids = self.prepare_batch(data_mega, cur_data_idx, encoding_num=self.trg_encoding_num)
                    batch.set_mega(*batch_info, trg_kb_ids)
                if data_mid is not None:
                    batch_info, mid_kb_ids = self.prepare_batch(data_mid, cur_data_idx, encoding_num=self.mid_encoding_num)
                    batch.set_mid(*batch_info, mid_kb_ids)
                # move to device
                batch.to(device)
                batches.append(batch)

            return batches

        # pad both source and target words
        def create_batches(self, dataset:str, is_src=None, is_mid=None) -> List[BaseBatch]:
            # self.train_mid could be None!
            if dataset == "train":
                batches = self.create_batch(dataset, self.train_src, self.train_trg, data_mid=self.train_mid)
            elif dataset == "dev":
                batches = self.create_batch(dataset, self.dev_src, self.dev_trg, data_mid=self.dev_mid)
            else:
                assert is_src is not None and is_mid is not None
                if is_mid:
                    batches = self.create_batch(dataset, data_src=None, data_trg=None, data_mega=None, data_mid=self.test_mid)
                else:
                    if is_src:
                        batches = self.create_batch(dataset, self.test_src, None, None, None)
                    else:
                        batches = self.create_batch(dataset, None, self.test_trg, None, None)

            return batches

        def create_megabatch(self, model:Encoder):
            # only for training
            data_src, data_trg = self.train_src, self.train_trg
            data_idx = [i for i in range(len(data_src))]
            random.shuffle(data_idx)
            for i in range(0, len(data_idx), self.mega_batch_size):
                batch = self.new_batch()
                cur_size = min(self.mega_batch_size, len(data_idx) - i)
                cur_data_idx = data_idx[i: i+cur_size]
                # src
                batch_info, src_gold_kb_ids = self.prepare_batch(data_src, cur_data_idx)
                batch.set_src(*batch_info, src_gold_kb_ids)
                # trg
                batch_info, trg_kb_ids = self.prepare_batch(data_trg, cur_data_idx)
                batch.set_trg(*batch_info, trg_kb_ids)

                with torch.no_grad():
                    model.eval()
                    batch.to(device)
                    M, _ = model.calc_batch_similarity(batch, use_negative=False, use_mid=False, proportion=0, trg_encoding_num=0, mid_encoding_num=0)
                    model.train()
                    raise NotImplementedError

                negative_num = min(1, cur_size - 1)

                # mask the non negative samples and the diagonal
                idx_tensor = torch.LongTensor(cur_data_idx)
                non_neg_mask = torch.index_select(self.non_neg_mask, 0, idx_tensor)
                non_neg_mask = torch.index_select(non_neg_mask, 1, idx_tensor)
                non_neg_mask = non_neg_mask.to(device)
                masked_M = M.masked_fill(non_neg_mask == 1, -1e9)

                # negative_idx = [batch_size, 1]
                _, negative_idx = torch.topk(masked_M, k=negative_num, dim=-1)
                # negative sample
                cur_negative_idx = [cur_data_idx[idx.item()] for idx in negative_idx]
                mega_src = [data_src[idx] for idx in cur_data_idx]
                mega_trg = [data_trg[idx] for idx in cur_data_idx]
                mega_negative = [data_trg[idx] for idx in cur_negative_idx]

                cur_mega_batch = self.create_batch("train", mega_src, mega_trg, mega_negative)
                for b in cur_mega_batch:
                    yield b

        def save_map(self, map, map_file):
            # save map
            with open(map_file, "wb") as f:
                pickle.dump(dict(map), f)
                print("[INFO] save x to idx map to :{}, len: {:d}".format(map_file, len(map)))

        def load_map(self, map_file):
            with open(map_file, "rb") as f:
                m = pickle.load(f)
                m = defaultdict(lambda: m[self.pad_str], m)
                print("[INFO] load x to idx map from {}, len: {:d}".format(map_file, len(m)))
                return m

def list2nparr(all_lists:List[List], hidden_size:int, merge=False):
    combined_encodings = []
    for version_list in all_lists:
        # last batch might not match the size
        encodings = np.concatenate(version_list, axis=0)
        combined_encodings.append(encodings)
    if merge:
        return np.vstack(combined_encodings)
    else:
        return combined_encodings

def calc_batch_loss(model, criterion, batch: BaseBatch, proportion, trg_encoding_num, mid_encoding_num):
    # src_tensor, src_lens, src_perm_idx, trg_tensor, trg_kb_id, trg_lens, trg_perm_idx
    similarity, diff = model.calc_batch_similarity(batch, use_negative=True, use_mid=True,
                                                   proportion=proportion, trg_encoding_num=trg_encoding_num, mid_encoding_num=mid_encoding_num)
    if diff is not None:
        loss = criterion(similarity) + diff
    else:
        loss = criterion(similarity)
    return loss

def get_unique_kb_idx(kb_id_list: list):
    find_kb_ids = []
    unique_kb_idx = []
    for i, id in enumerate(kb_id_list):
        if id in find_kb_ids:
            continue
        else:
            unique_kb_idx.append(i)
            find_kb_ids.append(id)
    return np.array(unique_kb_idx)

def append_multiple_encodings(encoding_list, encodings, encoding_num):
    diff_encodings = np.vsplit(encodings, encoding_num)
    for idx, e in enumerate(diff_encodings):
        encoding_list[idx].append(e)

def merge_encodings(encoding1:List[np.ndarray], encoding2:List[np.ndarray]):
    assert len(encoding1) == len(encoding2)
    combined_encodings = [np.vstack([x, y]) for x, y in zip(encoding1, encoding2)]
    combined_encodings = np.vstack(combined_encodings)
    return combined_encodings

# evaluate the whole dataset
def eval_data(model: Encoder, train_batches:List[BaseBatch], dev_batches: List[BaseBatch], similarity_measure: Similarity, args_dict: dict):
    use_mid = args_dict["use_mid"]
    topk = args_dict["topk"]
    trg_encoding_num = args_dict["trg_encoding_num"]
    mid_encoding_num = args_dict["mid_encoding_num"]
    # treat train target strings as the KB
    recall = 0
    tot = 0
    KB_encodings = [[] for _ in range(trg_encoding_num)]
    KB_ids = []
    for batch in train_batches:
        cur_encodings = np.array(model.calc_encode(batch, is_src=False).cpu())
        append_multiple_encodings(KB_encodings, cur_encodings, trg_encoding_num)
        KB_ids += batch.trg_kb_ids
    assert len(KB_encodings[0]) == len(train_batches)
    KB_encodings = list2nparr(KB_encodings, model.hidden_size)

    src_encodings = []
    trg_encodings = [[] for _ in range(trg_encoding_num)]
    trg_kb_ids = []
    for batch in dev_batches:
        src_encodings.append(np.array(model.calc_encode(batch, is_src=True).cpu()))
        cur_encodings = np.array(model.calc_encode(batch, is_src=False).cpu())
        append_multiple_encodings(trg_encodings, cur_encodings, trg_encoding_num)
        trg_kb_ids += batch.trg_kb_ids
    assert len(src_encodings) == len(dev_batches)
    assert len(trg_encodings[0]) == len(dev_batches)

    src_encodings = list2nparr([src_encodings], model.hidden_size, merge=True)
    trg_encodings = list2nparr(trg_encodings, model.hidden_size)


    # TODO might need it in the future
    # prune KB_encodings so that all entities are unique
    # unique_kb_idx = get_unique_kb_idx(KB_ids)
    # KB_encodings = KB_encodings[unique_kb_idx]

    all_trg_encodings = merge_encodings(trg_encodings, KB_encodings)
    n = max(all_trg_encodings.shape[0], 160000)
    all_trg_encodings = all_trg_encodings[:n]
    # calculate similarity`
    # [dev_size, dev_size + kb_size]
    scores = similarity_measure(src_encodings, all_trg_encodings, model.bilinear, split=True, pieces=10, negative_sample=None, encoding_num=trg_encoding_num)
    encoding_scores = np.copy(scores)
    if use_mid:
        mid_KB_encodings = [[] for _ in range(mid_encoding_num)]
        for batch in train_batches:
            cur_encodings = np.array(model.calc_encode(batch, is_src=False, is_mid=True).cpu())
            append_multiple_encodings(mid_KB_encodings, cur_encodings, mid_encoding_num)
            KB_ids += batch.trg_kb_ids
        assert len(mid_KB_encodings[0]) == len(train_batches)
        mid_KB_encodings = list2nparr(mid_KB_encodings, model.hidden_size)

        mid_encodings = [[] for _ in range(mid_encoding_num)]
        for batch in dev_batches:
            cur_encodings = np.array(model.calc_encode(batch, is_src=False, is_mid=True).cpu())
            append_multiple_encodings(mid_encodings, cur_encodings, mid_encoding_num)
        assert len(mid_encodings[0]) == len(dev_batches)
        mid_encodings = list2nparr(mid_encodings, model.hidden_size)

        # TODO might need it in the future
        # mid_KB_encodings = mid_KB_encodings[unique_kb_idx]
        all_mid_encodings = merge_encodings(mid_encodings, mid_KB_encodings)
        all_mid_encodings = all_mid_encodings[:n]
        all_mid_encodings = all_mid_encodings[:n]

        mid_scores = similarity_measure(src_encodings, all_mid_encodings, model.bilinear_mid, split=True, pieces=10, negative_sample=None, encoding_num=mid_encoding_num)
        scores = np.maximum(scores, mid_scores)

    for entry_idx, entry_scores in enumerate(scores):
        ranked_idxes = entry_scores.argsort()[::-1]
        # the correct index is entry_idx
        if entry_idx in ranked_idxes[:topk]:
            recall += 1
        tot += 1

    recall_2 = 0
    for entry_idx, entry_scores in enumerate(encoding_scores):
        ranked_idxes = entry_scores.argsort()[::-1]
        # the correct index is entry_idx
        if entry_idx in ranked_idxes[:topk]:
            recall_2 += 1

    return [recall, recall_2], tot

def reset_bias(module):
    param = module.state_dict()["bias_hh_l0"]
    p = torch.zeros_like(param)
    p[512:1024] = 1.0
    param.copy_(p)

    param = module.state_dict()["bias_hh_l0_reverse"]
    p = torch.zeros_like(param)
    p[512:1024] = 1.0
    param.copy_(p)

def run(data_loader: BaseDataLoader, encoder: Encoder, criterion, optimizer: optim, scheduler: optim.lr_scheduler,
          similarity_measure: Similarity, save_model,
          args:argparse.Namespace):
    encoder.to(device)
    best_accs = {"encode_acc": float('-inf'), "pivot_acc": float('-inf')}
    last_update = 0
    dev_arg_dict = {
        "use_mid": args.use_mid,
        "topk": 30,
        "trg_encoding_num": args.trg_encoding_num,
        "mid_encoding_num": args.mid_encoding_num
    }
    lr_decay = scheduler is not None
    if lr_decay:
        print("[INFO] using learning rate decay")
    for ep in range(args.max_epoch):
        encoder.train()
        train_loss = 0.0
        start_time = time.time()
        if not args.mega:
            train_batches = data_loader.create_batches("train")
        else:
            if ep <= 30:
                train_batches = data_loader.create_batches("train")
            else:
                train_batches = data_loader.create_megabatch(encoder)
        batch_num = 0
        for idx, batch in enumerate(train_batches):
            optimizer.zero_grad()
            cur_loss = calc_batch_loss(encoder, criterion, batch, args.mid_proportion, args.trg_encoding_num, args.mid_encoding_num)
            train_loss += cur_loss.item()
            cur_loss.backward()
            # optimizer.step()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            optimizer.step()

            if encoder.name == "bilstm":
                # set all but forget gate bias to 0
                reset_bias(encoder.src_lstm)
                reset_bias(encoder.trg_lstm)

            batch_num += 1
        print("[INFO] epoch {:d}: train loss={:.8f}, time={:.2f}".format(ep, train_loss / batch_num,
                                                                         time.time()-start_time))

        if (ep + 1) % EPOCH_CHECK == 0:
            with torch.no_grad():
                encoder.eval()
                # eval
                train_batches = data_loader.create_batches("train")
                dev_batches = data_loader.create_batches("dev")
                start_time = time.time()

                recall, tot = eval_data(encoder, train_batches, dev_batches, similarity_measure, dev_arg_dict)
                dev_pivot_acc = recall[0] / float(tot)
                dev_encode_acc = recall[1] / float(tot)
                if dev_encode_acc > best_accs["encode_acc"]:
                    best_accs["encode_acc"] = dev_encode_acc
                    best_accs["pivot_acc"] = dev_pivot_acc
                    last_update = ep + 1
                    save_model(encoder, ep + 1, train_loss, optimizer, args.model_path + "_" + "best" + ".tar")
                save_model(encoder, ep + 1, train_loss, optimizer, args.model_path + "_" + "last" + ".tar")
                print("[INFO] epoch {:d}: encoding/pivoting dev acc={:.4f}/{:.4f}, time={:.2f}".format(
                                                                                            ep, dev_encode_acc, dev_pivot_acc,
                                                                                            time.time()-start_time))
                if ep + 1 - last_update > PATIENT:
                    print("[FINAL] in epoch {}, the best develop encoding/pivoting accuracy = {:.4f}/{:.4f}".format(ep + 1,
                                                                                                                    best_accs["encode_acc"],
                                                                                                                    best_accs["pivot_acc"]))
                    break
        if lr_decay:
            scheduler.step()

def init_train(args, DataLoader):
    train_file = FileInfo()
    train_file.set_all(args.train_file, args.src_idx, args.trg_idx, args.trg_id_idx, args.trg_type_idx)
    train_file.set_mid(args.train_mid_file, args.mid_str_idx, args.mid_id_idx, args.mid_type_idx)
    dev_file = FileInfo()
    dev_file.set_all(args.dev_file, args.src_idx, args.trg_idx, args.trg_id_idx, args.trg_type_idx)
    dev_file.set_mid(args.dev_mid_file, args.mid_str_idx, args.mid_id_idx, args.mid_type_idx)
    data_loader = DataLoader(True, args.map_file, args.batch_size, args.mega_size, args.use_panphon, args.use_mid, args.share_vocab, train_file=train_file,
                             dev_file=dev_file, test_file=None, trg_encoding_num=args.trg_encoding_num, mid_encoding_num=args.mid_encoding_num)
    similarity_measure = Similarity(args.similarity_measure)

    if args.objective == "hinge":
        criterion = MultiMarginLoss(device, margin=args.margin, reduction="mean")
        # criterion = MultiMarginLoss(device, margin=args.margin, reduction="sum")
    elif args.objective == "mle":
        criterion = CrossEntropyLoss(device, reduction="mean")
    else:
        raise NotImplementedError

    return data_loader, criterion, similarity_measure

def create_optimizer(trainer, lr, model, lr_decay=False):
    if trainer == "adam":
        optimizer = optim.Adam(model.parameters(), lr)
    elif trainer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr)
    else:
        raise NotImplementedError
    if lr_decay:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 50, 100], gamma=0.3)
    else:
        scheduler = None
    return optimizer, scheduler

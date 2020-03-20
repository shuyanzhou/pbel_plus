import sys
import os
import functools
import random
import numpy as np
import pickle
from typing import List
from collections import defaultdict, Counter
from utils.constant import DEVICE
from utils.func import FileInfo

print = functools.partial(print, flush=True)
device = DEVICE

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

class BaseDataLoader:
    def __init__(self, is_train, args,
                 train_file: FileInfo, dev_file: FileInfo, test_file: FileInfo):
        self.batch_size = args.batch_size
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.map_file = args.map_file
        self.pad_str = "<UNK>"
        self.pad_idx = 0
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.use_mid = args.use_mid
        self.trg_encoding_num = args.trg_encoding_num
        self.mid_encoding_num = args.mid_encoding_num
        self.load_alia_map(args.alia_file)
        self.n_gram_threshold = args.n_gram_threshold
        self.max_position = 0
        if is_train:
            self.init_train()
        else:
            self.init_test()

    def n_gram_filter(self, data, freq_map):
        filter_data = []
        for cur_data in data:
            all_string_idx = cur_data[0][0]
            cur_filter_string, cur_filter_st, cur_filter_ed = [], [], []
            for cur_version, cur_version_string in enumerate(all_string_idx):
                filter_idx = []
                for idx, ngram_idx in enumerate(cur_version_string):
                    if ngram_idx == self.pad_idx or freq_map[idx] < self.n_gram_threshold:
                        continue
                    else:
                        filter_idx.append(idx)
                if len(filter_idx) != 0:
                    filter_string = [cur_version_string[x] for x in filter_idx]
                else:
                    filter_string = [self.pad_idx]

                cur_filter_string.append(filter_string)
            all_info = [cur_filter_string]
            filter_data.append([all_info, cur_data[1]])

        return filter_data

    def init_train(self):
        self.x2i_src = defaultdict(lambda: len(self.x2i_src))
        self.x2i_trg = defaultdict(lambda: len(self.x2i_trg))
        # make sure pad is 0
        self.x2i_src[self.pad_str]
        self.x2i_trg[self.pad_str]
        self.src_freq_map = Counter()
        self.trg_freq_map = Counter()
        self.train_src = list(self.load_data(self.train_file.src_file_name, self.train_file.src_str_idx,
                                             self.train_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
        self.train_trg = list(self.load_data(self.train_file.trg_file_name, self.train_file.trg_str_idx,
                                             self.train_file.trg_id_idx, is_src=False,
                                             encoding_num=self.trg_encoding_num,
                                             type_idx=self.train_file.trg_type_idx))
        # save map
        self.save_map(self.x2i_src, self.map_file + "_src.pkl")
        self.save_map(self.x2i_trg, self.map_file + "_trg.pkl")
        self.save_map(self.src_freq_map, self.map_file + "_src_freq.pkl")
        self.save_map(self.trg_freq_map, self.map_file + "_trg_freq.pkl")

        if self.use_mid:
            self.x2i_mid = defaultdict(lambda: len(self.x2i_mid))
            self.x2i_mid[self.pad_str]
            self.mid_freq_map = Counter()
            self.train_mid = list(
                self.load_data(self.train_file.mid_file_name, self.train_file.mid_str_idx, self.train_file.mid_id_idx,
                               is_src=False, is_mid=True, encoding_num=self.mid_encoding_num,
                               type_idx=self.train_file.mid_type_idx))
            self.save_map(self.x2i_mid, self.map_file + "_mid.pkl")
            self.save_map(self.mid_freq_map, self.map_file + "_mid_freq.pkl")
            self.mid_vocab_size = len(self.x2i_mid)
            self.x2i_mid = defaultdict(lambda: self.x2i_mid[self.pad_str], self.x2i_mid)
            self.mid_freq_map = defaultdict(lambda: float('-inf'), self.mid_freq_map)
        else:
            self.train_mid = None
            self.mid_vocab_size = 0

        self.src_vocab_size = len(self.x2i_src)
        self.trg_vocab_size = len(self.x2i_trg)
        self.x2i_src = defaultdict(lambda: self.x2i_src[self.pad_str], self.x2i_src)
        self.x2i_trg = defaultdict(lambda: self.x2i_trg[self.pad_str], self.x2i_trg)
        self.src_freq_map = defaultdict(lambda: float('-inf'), self.src_freq_map)
        self.trg_freq_map = defaultdict(lambda: float('-inf'), self.trg_freq_map)

        if self.dev_file:
            self.dev_src = list(self.load_data(self.dev_file.src_file_name, self.dev_file.src_str_idx,
                                               self.dev_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
            self.dev_trg = list(self.load_data(self.dev_file.trg_file_name, self.dev_file.trg_str_idx,
                                               self.dev_file.trg_id_idx, is_src=False,
                                               encoding_num=self.trg_encoding_num, type_idx=self.dev_file.trg_type_idx))
            n = min(len(self.dev_src), 2000)
            self.dev_src, self.dev_trg = self.dev_src[:n], self.dev_trg[:n]
            if self.use_mid:
                self.dev_mid = list(self.load_data(self.dev_file.mid_file_name, self.dev_file.mid_str_idx,
                                                   self.dev_file.mid_id_idx, is_src=False, is_mid=True,
                                                   encoding_num=self.mid_encoding_num,
                                                   type_idx=self.dev_file.mid_type_idx))
                self.dev_mid = self.dev_mid[:n]
            else:
                self.dev_mid = None
        else:
            self.dev_src, self.dev_trg, self.dev_mid = None, None, None

        if self.n_gram_threshold != 0:
            self.train_src = self.n_gram_filter(self.train_src, self.src_freq_map)
            self.train_trg = self.n_gram_filter(self.train_trg, self.trg_freq_map)
            if self.use_mid:
                self.train_mid = self.n_gram_filter(self.train_mid, self.mid_freq_map)

            # recover from training frequency
            if self.dev_file:
                self.dev_src = self.n_gram_filter(self.dev_src, self.src_freq_map)
                self.dev_trg = self.n_gram_filter(self.dev_trg, self.trg_freq_map)
                if self.use_mid:
                    self.dev_mid = self.n_gram_filter(self.dev_mid, self.mid_freq_map)


    def init_test(self):
        self.x2i_src = self.load_map(self.map_file + "_src.pkl")
        self.x2i_trg = self.load_map(self.map_file + "_trg.pkl")
        self.src_freq_map = self.load_map(self.map_file + "_src_freq.pkl", float('-inf')) if os.path.exists(
            self.map_file + "_src_freq.pkl") \
            else defaultdict(int)
        self.trg_freq_map = self.load_map(self.map_file + "_trg_freq.pkl", float('-inf')) if os.path.exists(
            self.map_file + "_trg_freq.pkl") \
            else defaultdict(int)
        if self.use_mid:
            self.x2i_mid = self.load_map(self.map_file + "_mid.pkl")
            self.mid_freq_map = self.load_map(self.map_file + "_mid_freq.pkl", float('-inf'))
        else:
            self.x2i_mid = None
        self.i2c_src = {v: k for k, v in self.x2i_src.items()}
        self.i2c_trg = {v: k for k, v in self.x2i_trg.items()}
        if self.test_file.src_file_name is not None:
            self.test_src = list(self.load_data(self.test_file.src_file_name,
                                                self.test_file.src_str_idx, self.test_file.src_id_idx,
                                                is_src=True, encoding_num=1, type_idx=None))
        if self.test_file.trg_file_name is not None:
            self.test_trg = list(self.load_data(self.test_file.trg_file_name,
                                                self.test_file.trg_str_idx, self.test_file.trg_id_idx,
                                                is_src=False, encoding_num=self.trg_encoding_num,
                                                type_idx=self.test_file.trg_type_idx))
        if self.test_file.mid_file_name is not None:
            self.test_mid = list(self.load_data(self.test_file.mid_file_name,
                                                self.test_file.mid_str_idx, self.test_file.mid_id_idx, is_src=False,
                                                is_mid=True,
                                                encoding_num=self.mid_encoding_num,
                                                type_idx=self.test_file.mid_type_idx))

        if self.n_gram_threshold != 0:
            if self.test_file.src_file_name is not None:
                self.test_src = self.n_gram_filter(self.test_src, self.src_freq_map)
            if self.test_file.trg_file_name is not None:
                self.test_trg = self.n_gram_filter(self.test_trg, self.trg_freq_map)
            if self.test_file.mid_file_name is not None:
                self.test_mid = self.n_gram_filter(self.test_mid, self.mid_freq_map)

    def load_alia_map(self, fname):
        if fname != "HOLDER":
            self.title_alia_map = defaultdict(list)
            self.id_alia_map = defaultdict(list)
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    tks = line.strip().split(" ||| ")
                    if len(tks) != 4:
                        continue
                    aka = tks[3].split(" || ")
                    self.title_alia_map[tks[2]] = aka
                    if tks[1] != "NAN":
                        self.id_alia_map[tks[1]] = aka
            print(f"[INFO] there are {len(self.title_alia_map)} / {len(self.id_alia_map)} items in aka")
        else:
            print("[WARNING] no alia file found!")

    def get_alias(self, tks, str_idx, id_idx, encoding_num):
        id = tks[id_idx]
        title = tks[str_idx]
        alias = self.title_alia_map.get(title, []) + self.id_alia_map.get(id, [])
        alias = [x for x in alias if x != title]

        if len(alias) < encoding_num:
            alias = [title for x in range(encoding_num - len(alias))] + alias
        # randomly select encoding num - 1 alias
        else:
            selected_idx = np.random.choice(len(alias), encoding_num - 1, replace=False)
            alias = [alias[x] for x in selected_idx]
            alias = [title] + alias

        assert len(alias) == encoding_num

        return alias

    def load_all_data(self, file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx):
        raise NotImplementedError

    def load_data(self, file_name, str_idx, id_idx, is_src, encoding_num, type_idx, is_mid=False):
        if is_src:
            x2i_map = self.x2i_src
            freq_map = self.src_freq_map
        else:
            x2i_map = self.x2i_trg
            freq_map = self.trg_freq_map
        if is_mid:
            x2i_map = self.x2i_mid
            freq_map = self.mid_freq_map
        return self.load_all_data(file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx)

    def transform_one_batch(self, *args, **kwargs) -> list:
        raise NotImplementedError

    def new_batch(self) -> BaseBatch:
        raise NotImplementedError

    def extract_idx(self, encoding_num, idx_list):
        all_idx = []
        for i in range(encoding_num):
            all_idx += [idx_list[idx][i] for idx in range(len(idx_list))]
        assert len(all_idx) == len(idx_list) * encoding_num

        return all_idx

    # data from one side
    def prepare_batch(self, side_data, data_idx, encoding_num):
        # this is a list of words
        words = [side_data[idx][0][0] for idx in data_idx]
        # expand words to list
        all_words = self.extract_idx(encoding_num, words)
        word_idx_tensor, *other_info = self.transform_one_batch(all_words)
        merge_tensor = word_idx_tensor

        kb_ids = [side_data[idx][1] for idx in data_idx]
        batch_info = [merge_tensor, *other_info]

        return batch_info, kb_ids

    def create_batch(self, dataset, data_src=None, data_trg=None, data_mid=None) -> List[BaseBatch]:
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
            if data_mid is not None:
                batch_info, mid_kb_ids = self.prepare_batch(data_mid, cur_data_idx, encoding_num=self.mid_encoding_num)
                batch.set_mid(*batch_info, mid_kb_ids)
            # move to device
            batch.to(device)
            batches.append(batch)

        return batches

    # pad both source and target words
    def create_batches(self, dataset: str, is_src=None, is_mid=None) -> List[BaseBatch]:
        # self.train_mid could be None!
        # training time
        if dataset == "train":
            batches = self.create_batch(dataset, self.train_src, self.train_trg, data_mid=self.train_mid)
        elif dataset == "dev":
            batches = self.create_batch(dataset, self.dev_src, self.dev_trg, data_mid=self.dev_mid)
        # test time, load data separately
        else:
            assert is_src is not None and is_mid is not None
            if is_mid:
                batches = self.create_batch(dataset, data_src=None, data_trg=None,
                                            data_mid=self.test_mid)
            else:
                if is_src:
                    batches = self.create_batch(dataset, self.test_src, None, None)
                else:
                    batches = self.create_batch(dataset, None, self.test_trg, None)

        return batches

    def save_map(self, map, map_file):
        # save map
        with open(map_file, "wb") as f:
            pickle.dump(dict(map), f)
            print("[INFO] save x to idx map to :{}, len: {:d}".format(map_file, len(map)))

    def load_map(self, map_file, default_return=None):
        with open(map_file, "rb") as f:
            m = pickle.load(f)
            if default_return is None:
                default_return = m[self.pad_str]
            m = defaultdict(lambda: default_return, m)
            print("[INFO] load x to idx map from {}, len: {:d}".format(map_file, len(m)))
            return m
import sys

sys.path.append("/home/shuyanzh/workshop/cmu_lorelei_edl/")
from collections import defaultdict
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import random
import math
import panphon as pp
from base_train import FileInfo, BaseBatch, BaseDataLoader, Encoder, init_train, create_optimizer, run
from base_test import init_test, eval_dataset, reset_unk_weight
from config import argps
from similarity_calculator import Similarity
from utils.constant import DEVICE, RANDOM_SEED
import numpy as np

random_seed = RANDOM_SEED
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
print = functools.partial(print, flush=True)
device = DEVICE


class Batch(BaseBatch):
    def set_src(self, src_tensor, src_mask, src_gold_kb_ids):
        self.src_tensor = src_tensor.long()
        self.src_mask = src_mask
        self.gold_kb_ids = src_gold_kb_ids
        self.src_flag = True

    def set_trg(self, trg_tensor, trg_mask, trg_kb_ids):
        self.trg_tensor = trg_tensor.long()
        self.trg_mask = trg_mask
        self.trg_kb_ids = trg_kb_ids
        self.trg_flag = True

    def set_mega(self, mega_tensor, mega_mask, mega_trg_kb_ids):
        self.mega_tensor = mega_tensor.long()
        self.mega_mask = mega_mask
        self.mega_trg_kb_ids = mega_trg_kb_ids
        self.mega_flag = True
        self.negative_num = 1

    def set_mid(self, mid_tensor, mid_mask, mid_kb_ids):
        self.mid_tensor = mid_tensor.long()
        self.mid_mask = mid_mask
        self.mid_kb_ids = mid_mask
        self.mid_flag = True

    def to(self, device):
        if self.src_flag:
            self.src_tensor = self.src_tensor.to(device)
            self.src_mask = self.src_mask.to(device)
        if self.trg_flag:
            self.trg_tensor = self.trg_tensor.to(device)
            self.trg_mask = self.trg_mask.to(device)
        if self.mega_flag:
            self.mega_tensor = self.mega_tensor.to(device)
            self.mega_mask = self.mega_mask.to(device)
        if self.mid_flag:
            self.mid_tensor = self.mid_tensor.to(device)
            self.mid_mask = self.mid_mask.to(device)

    def get_all(self):
        return self.src_tensor, self.src_mask, \
               self.trg_tensor, self.trg_mask

    def get_src(self):
        return self.src_tensor, self.src_mask

    def get_trg(self):
        return self.trg_tensor, self.trg_mask

    def get_mega(self):
        return self.mega_tensor, self.mega_mask

    def get_mid(self):
        return self.mid_tensor, self.mid_mask


class DataLoader(BaseDataLoader):
    def __init__(self, is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab,
                 train_file, dev_file, test_file,
                 trg_encoding_num, mid_encoding_num, trg_auto_encoding, mid_auto_encoding, alia_file, n_gram_threshold,
                 position_embedding):
        super(DataLoader, self).__init__(is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab,
                                         "<UNK>",
                                         train_file, dev_file, test_file,
                                         trg_encoding_num, mid_encoding_num,
                                         trg_auto_encoding, mid_auto_encoding, alia_file, n_gram_threshold,
                                         position_embedding)

    def new_batch(self):
        return Batch()

    def load_all_data(self, file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx, auto_encoding):
        line_tot = 0
        with open(file_name, "r", encoding="utf-8") as fin:
            for line in fin:
                line_tot += 1
                tks = line.strip().split(" ||| ")
                mention_string = "<" +  tks[str_idx] + ">"
                if encoding_num == 1:
                    string = [x2i_map[x] for x in mention_string]
                    all_string = [string]
                    all_st = [0 for x in range(len(string))]
                    all_ed = [0 for x in range(len(string))]
                else:
                    if auto_encoding:
                        all_string = []
                        all_st = []
                        all_ed = []
                        for i in range(encoding_num):
                            string = [x2i_map[x] for x in
                                      ["<" + tks[type_idx] + ">"] + ["<" + str(i) + ">"] + mention_string]
                            all_string.append(string)
                    else:
                        all_string = []
                        all_st = []
                        all_ed = []
                        alias = self.get_alias(tks, str_idx, id_idx, encoding_num)
                        for i in range(encoding_num):
                            cur_mention_string = "<" + alias[i] + ">"
                            string = [x2i_map[x] for x in cur_mention_string]
                            all_string.append(string)

                for s in all_string:
                    for ss in s:
                        freq_map[ss] += 1


                if self.position_embedding:
                    raise NotImplementedError
                else:
                    all_info = [all_string]

                yield (all_info, tks[id_idx])

        print("[INFO] number of lines in {}: {}".format(file_name, str(line_tot)))

    def transform_one_batch(self, data):
        data_len = [len(x) for x in data]
        cur_size = len(data_len)
        max_data_len = max(data_len)
        data_tensor = torch.zeros((cur_size, max_data_len))
        for idx, id_list in enumerate(data):
            data_tensor[idx, :data_len[idx]] = torch.LongTensor(id_list)
        mask = (data_tensor != self.pad_idx)
        return [data_tensor, mask]


class CharCNN(Encoder):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, similarity_measure, use_mid,
                 share_vocab, position_embedding, max_position, st_weight, ed_weight, sin_embedding, mid_vocab_size=0):
        super(CharCNN, self).__init__(embed_size)
        self.name = "charcnn"
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.mid_vocab_size = mid_vocab_size
        self.use_mid = use_mid
        self.share_vocab = share_vocab
        self.embed_size = embed_size
        self.position_embedding = position_embedding
        self.max_position = 0
        self.hidden_size = 300
        self.sin_embedding = sin_embedding
        self.st_weight, self.ed_weight = st_weight, ed_weight
        self.activate = F.relu
        # parameters
        self.src_lookup = nn.Embedding(src_vocab_size, embed_size, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.src_lookup.weight, gain=1)
        self.trg_lookup = nn.Embedding(trg_vocab_size, embed_size, padding_idx=0)
        torch.nn.init.xavier_uniform_(self.trg_lookup.weight, gain=1)

        if position_embedding:
            raise NotImplementedError


        if use_mid:
            raise NotImplementedError

        self.window_size =  [2,3,4,5]
        self.padding = [0, 1, 0, 2]
        self.src_conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=self.hidden_size, kernel_size=ws,
                                 stride=1, padding=pd, dilation=1, groups=1, bias=True) for pd, ws in zip(self.padding, self.window_size)])
        self.trg_conv1d_list = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=self.hidden_size, kernel_size=ws,
                                 stride=1, padding=pd, dilation=1, groups=1, bias=True) for pd, ws in zip(self.padding, self.window_size)])

        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.hidden_size * len(self.window_size), self.hidden_size)
        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.similarity_measure = similarity_measure

    def assign_weight(self, lookup, weight):
        lookup.weight = nn.Parameter(weight, requires_grad=False)

    # calc_batch_similarity will return the similarity of the batch
    # while calc encode only return the encoding result of src or trg of the batch
    def calc_encode(self, batch: Batch, is_src, is_mega=False, is_mid=False):
        # input: [len, batch] or [len, batch, pp_vec_size]
        # embed: [len, batch, embed_size]

        if is_mid:
           raise NotImplementedError
        else:
            if is_src:
                lookup = self.src_lookup
                conv1d_list = self.src_conv1d_list
                input, mask = batch.get_src()
                if self.position_embedding:
                    raise NotImplementedError
            else:
                lookup = self.trg_lookup
                conv1d_list = self.trg_conv1d_list

                if self.position_embedding:
                    raise NotImplementedError
                if is_mega:
                    raise NotImplementedError
                else:
                    input, mask = batch.get_trg()

        # will contain 3 part
        if self.position_embedding:
            raise NotImplementedError

        else:
            # get masks
            all_masks = []
            for pd, ws in zip(self.padding, self.window_size):
                if pd == 0:
                    cur_mask = mask[:, ws - 1:]
                    all_masks.append(cur_mask.unsqueeze(1).float())
                else:
                    all_masks.append(mask.unsqueeze(1).float())

            # [batch_size, max_len, embed_size]
            embed = lookup(input)
            # [batch_size, embed_size, max_len]
            reshape_embed = torch.transpose(embed, 1, 2)
            # [batch_size, hidden_size, length - window_size + 1] * len(window_size)
            conv_result_list = [conv1d_layer(reshape_embed) for conv1d_layer in conv1d_list]
            conv_result_list = [self.activate(result) for result in conv_result_list]
            masked_conv_result_list = [(x - (1 - cur_mask) * 1e10) for cur_mask, x in zip(all_masks, conv_result_list)]
            # [batch_size, hidden_size]
            max_pooling_list = [F.max_pool1d(conv_result, kernel_size=conv_result.size(2)).squeeze(2) for conv_result in masked_conv_result_list]
            dropout_list = [self.dropout(max_pooling) for max_pooling in max_pooling_list]
            concat = torch.cat(dropout_list, dim=1)
            encode = self.linear(concat)
            # [batch_size, hidden_size, len(window_size)]
            # concat = torch.stack(dropout_list, dim=2)
            # encode = torch.sum(concat, dim=-1, keepdim=False)
        return encode


def save_model(model: CharCNN, epoch, loss, optimizer, model_path):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_statte_dict": optimizer.state_dict(),
                "src_vocab_size": model.src_vocab_size,
                "trg_vocab_size": model.trg_vocab_size,
                "mid_vocab_size": model.mid_vocab_size,
                "embed_size": model.embed_size,
                "similarity_measure": model.similarity_measure.method,
                "max_position": model.max_position,
                "sin_embedding": model.sin_embedding,
                "epoch": epoch,
                "loss": loss}, model_path)
    print("[INFO] save model!")


if __name__ == "__main__":
    args = argps()
    if args.is_train:
        data_loader, criterion, similarity_measure = init_train(args, DataLoader)
        model = CharCNN(data_loader.src_vocab_size, data_loader.trg_vocab_size,
                          args.embed_size, similarity_measure, args.use_mid, args.share_vocab,
                          position_embedding=args.position_embedding,
                          max_position=args.max_position,
                          st_weight=args.st_weight,
                          ed_weight=args.ed_weight,
                          sin_embedding=args.sin_embedding,
                          mid_vocab_size=data_loader.mid_vocab_size)
        optimizer, scheduler = create_optimizer(args.trainer, args.learning_rate, model)

        if args.finetune:
            model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
            model.load_state_dict(model["model_state_dict"])
            optimizer.load_state_dict(model["optimizer_state_dict"])
            print(
                "[INFO] load model from epoch {:d} train loss: {:.4f}".format(model_info["epoch"], model_info["loss"]))
        model.set_similarity_matrix()
        run(data_loader, model, criterion, optimizer, scheduler, similarity_measure, save_model, args)
    else:
        base_data_loader, intermedia_stuff = init_test(args, DataLoader)
        model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
        similarity_measure = Similarity(args.similarity_measure)
        model = CharCNN(model_info["src_vocab_size"], model_info["trg_vocab_size"],
                          model_info["embed_size"],
                          similarity_measure=similarity_measure,
                          use_mid=args.use_mid,
                          share_vocab=args.share_vocab,
                          position_embedding=args.position_embedding,
                          max_position=args.max_position,
                          st_weight=args.st_weight,
                          ed_weight=args.ed_weight,
                          sin_embedding=model_info.get("sin_embedding", False),
                          mid_vocab_size=model_info.get("mid_vocab_size", 0))
        model.load_state_dict(model_info["model_state_dict"])
        reset_unk_weight(model)
        model.set_similarity_matrix()
        eval_dataset(model, similarity_measure, base_data_loader, args.encoded_test_file, args.load_encoded_test,
                     args.encoded_kb_file, args.load_encoded_kb, intermedia_stuff, args.method, args.trg_encoding_num,
                     args.mid_encoding_num,
                     args.result_file, args.record_recall)

import sys
sys.path.append("/home/shuyanzh/workshop/cmu_lorelei_edl/")
from collections import defaultdict
import functools
import torch
from torch import nn
from torch import optim
import random
import panphon as pp
from  torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from base_train import FileInfo, BaseBatch, BaseDataLoader, Encoder, init_train, create_optimizer, run
from base_test import init_test, eval_dataset
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

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"

def get_ngram(string, ngram_list=(2, 3, 4, 5)):
    all_ngrams = []
    all_st_idx = []
    all_ed_idx = []
    char_list = [START_SYMBOL] + list(string) + [END_SYMBOL]
    for n in ngram_list:
        cur_ngram = zip(*[char_list[i:] for i in range(n)])
        cur_ngram = ["".join(x) for x in cur_ngram]
        all_ngrams += cur_ngram

        idx_list = [i for i in range(len(char_list))]
        cur_idx_ngram = zip(*[idx_list[i:] for i in range(n)])
        st_ed = [[x[0], x[-1]] for x in cur_idx_ngram]
        all_st_idx += [x[0] for x in st_ed]
        all_ed_idx += [x[1] for x in st_ed]
    return all_ngrams, all_st_idx, all_ed_idx


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
        self.mid_kb_ids= mid_mask
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
        return  self.src_tensor, self.src_mask, \
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
        super(DataLoader,self).__init__(is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab, "<UNK>",
                                        train_file, dev_file, test_file,
                                        trg_encoding_num, mid_encoding_num,
                                        trg_auto_encoding, mid_auto_encoding, alia_file, n_gram_threshold, position_embedding)

    def new_batch(self):
        return Batch()

    def load_all_data(self, file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx, auto_encoding):
        line_tot = 0
        with open(file_name, "r", encoding="utf-8") as fin:
            for line in fin:
                line_tot += 1
                tks = line.strip().split(" ||| ")
                if encoding_num == 1:
                    all_n_gram, st, ed = get_ngram(tks[str_idx])
                    string = [x2i_map[ngram] for ngram in all_n_gram]
                    all_string = [string]
                    all_st = [st]
                    all_ed = [ed]
                else:
                    if auto_encoding:
                        all_string = []
                        all_st = []
                        all_ed = []
                        for i in range(encoding_num):
                            all_n_gram, st, ed = get_ngram(tks[str_idx])
                            string = [x2i_map[ngram] for ngram in ["<" + tks[type_idx] + ">"] +  ["<" + str(i) + ">"] + all_n_gram]
                            all_string.append(string)
                            all_st.append(st)
                            all_ed.append(ed)
                    else:
                        all_string = []
                        all_st = []
                        all_ed = []
                        alias = self.get_alias(tks, str_idx, id_idx, encoding_num)
                        for i in range(encoding_num):
                            all_n_gram, st, ed = get_ngram(alias[i])
                            string = [x2i_map[ngram] for ngram in all_n_gram]
                            all_string.append(string)
                            all_st.append(st)
                            all_ed.append(ed)
                for s in all_string:
                    for ss in s:
                        freq_map[ss] += 1

                self.max_position = max(self.max_position, max([y for x in all_ed for y in x]))

                if self.position_embedding:
                    all_info = [all_string, all_st, all_ed]
                    for x, y, z in zip(all_string, all_st, all_ed):
                        assert len(x) == len(y) and len(y) == len(z)
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
        mask = (data_tensor != self.pad_idx).unsqueeze(-1)
        return [data_tensor, mask]
    
class Charagram(Encoder):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, similarity_measure, use_mid,
                 share_vocab, position_embedding, max_position, st_weight, ed_weight, mid_vocab_size=0):
        super(Charagram, self).__init__(embed_size)
        self.name = "charagram"
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.mid_vocab_size = mid_vocab_size
        self.use_mid = use_mid
        self.share_vocab = share_vocab
        self.hidden_size = embed_size
        self.embed_size = embed_size
        self.position_embedding = position_embedding
        self.max_position = max_position
        self.st_weight, self.ed_weight = st_weight, ed_weight
        self.activate = torch.tanh
        # parameters
        self.src_lookup = nn.Embedding(src_vocab_size, embed_size)
        torch.nn.init.xavier_uniform_(self.src_lookup.weight, gain=1)
        self.trg_lookup = nn.Embedding(trg_vocab_size, embed_size)
        torch.nn.init.xavier_uniform_(self.trg_lookup.weight, gain=1)


        if position_embedding:
            self.src_st_lookup = nn.Embedding(max_position, embed_size)
            self.src_ed_lookup = nn.Embedding(max_position, embed_size)
            self.trg_st_lookup = nn.Embedding(max_position, embed_size)
            self.trg_ed_lookup = nn.Embedding(max_position, embed_size)
            torch.nn.init.xavier_uniform_(self.src_st_lookup.weight, gain=1)
            torch.nn.init.xavier_uniform_(self.src_ed_lookup.weight, gain=1)
            torch.nn.init.xavier_uniform_(self.trg_st_lookup.weight, gain=1)
            torch.nn.init.xavier_uniform_(self.trg_ed_lookup.weight, gain=1)

        self.bias_src = nn.Parameter(torch.zeros(1, embed_size), requires_grad=True)
        self.bias_trg = nn.Parameter(torch.zeros(1, embed_size), requires_grad=True)
        torch.nn.init.constant_(self.bias_src, 0.0)
        torch.nn.init.constant_(self.bias_trg, 0.0)

        if use_mid:
            if share_vocab:
                self.mid_lookup = self.src_lookup
                self.bias_mid = self.bias_src
                if self.position_embedding:
                    self.mid_st_lookup = self.src_st_lookup
                    self.mid_ed_lookup = self.src_ed_lookup
            else:
                self.mid_lookup = nn.Embedding(mid_vocab_size, embed_size)
                torch.nn.init.xavier_uniform_(self.mid_lookup.weight, gain=1)
                if self.position_embedding:
                    self.mid_st_lookup = nn.Embedding(max_position, embed_size)
                    self.mid_ed_lookup = nn.Embedding(max_position, embed_size)
                    torch.nn.init.xavier_uniform_(self.mid_st_lookup.weight, gain=1)
                    torch.nn.init.xavier_uniform_(self.mid_ed_lookup.weight, gain=1)

                self.bias_mid = nn.Parameter(torch.zeros(1, embed_size), requires_grad=True)
                torch.nn.init.xavier_uniform_(self.bias_mid, gain=1)

        self.similarity_measure = similarity_measure


    # calc_batch_similarity will return the similarity of the batch
    # while calc encode only return the encoding result of src or trg of the batch
    def calc_encode(self, batch: Batch, is_src, is_mega=False, is_mid=False):
        # input: [len, batch] or [len, batch, pp_vec_size]
        # embed: [len, batch, embed_size]

        if is_mid:
            lookup = self.mid_lookup
            bias = self.bias_mid
            input, mask = batch.get_mid()
            if self.position_embedding:
                st_lookup = self.mid_st_lookup
                ed_lookup = self.mid_ed_lookup
        else:
            if is_src:
                lookup = self.src_lookup
                bias = self.bias_src
                input, mask = batch.get_src()
                if self.position_embedding:
                    st_lookup = self.src_st_lookup
                    ed_lookup = self.src_ed_lookup
            else:
                lookup = self.trg_lookup
                if self.position_embedding:
                    st_lookup = self.trg_st_lookup
                    ed_lookup = self.trg_ed_lookup
                if is_mega:
                    bias = self.bias_trg
                    input, mask = batch.get_mega()
                else:
                    bias = self.bias_trg
                    input, mask = batch.get_trg()

        # will contain 3 part
        if self.position_embedding:
            word_input, st_input, ed_input = torch.unbind(input, dim=-1)
            word_embed = lookup(word_input)
            st_embed = st_lookup(st_input)
            ed_embed = ed_lookup(ed_input)
            embed = word_embed + self.st_weight * st_embed + self.ed_weight * ed_embed
            embed = embed.masked_fill(mask==0, 0)
            encoded = self.activate(torch.sum(embed, dim=1, keepdim=False) + bias)

        else:
            # [batch_size, max_len, embed_size]
            embed = lookup(input)
            # mask padding
            embed = embed.masked_fill(mask==0, 0)
            # [batch_size, embed_size]
            encoded = self.activate(torch.sum(embed, dim=1, keepdim=False) + bias)
        return encoded

def save_model(model:Charagram, epoch, loss, optimizer, model_path):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_statte_dict": optimizer.state_dict(),
                "src_vocab_size": model.src_vocab_size,
                "trg_vocab_size": model.trg_vocab_size,
                "mid_vocab_size": model.mid_vocab_size,
                "embed_size": model.embed_size,
                "similarity_measure": model.similarity_measure.method,
                "max_position": model.max_position,
                "epoch": epoch,
                "loss": loss}, model_path)
    print("[INFO] save model!")

if __name__ == "__main__":
    args = argps()
    if args.is_train:
        data_loader, criterion, similarity_measure = init_train(args, DataLoader)
        model = Charagram(data_loader.src_vocab_size, data_loader.trg_vocab_size,
                    args.embed_size, similarity_measure, args.use_mid, args.share_vocab,
                          position_embedding=args.position_embedding,
                          max_position=data_loader.max_position + 1,
                          st_weight=args.st_weight,
                          ed_weight=args.ed_weight,
                          mid_vocab_size=data_loader.mid_vocab_size)
        optimizer, scheduler = create_optimizer(args.trainer, args.learning_rate, model)

        if args.finetune:
            model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
            model.load_state_dict(model["model_state_dict"])
            optimizer.load_state_dict(model["optimizer_state_dict"])
            print("[INFO] load model from epoch {:d} train loss: {:.4f}".format(model_info["epoch"], model_info["loss"]))
        model.set_similarity_matrix()
        run(data_loader, model, criterion, optimizer, scheduler, similarity_measure, save_model, args)
    else:
        base_data_loader, intermedia_stuff = init_test(args, DataLoader)
        model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
        similarity_measure = Similarity(args.similarity_measure)
        model = Charagram(model_info["src_vocab_size"], model_info["trg_vocab_size"],
                        model_info["embed_size"],
                        similarity_measure=similarity_measure,
                        use_mid=args.use_mid,
                        share_vocab=args.share_vocab,
                        position_embedding=args.position_embedding,
                        max_position=model_info.get("max_position", 0),
                        st_weight=args.st_weight,
                        ed_weight=args.ed_weight,
                        mid_vocab_size=model_info.get("mid_vocab_size", 0))
        model.load_state_dict(model_info["model_state_dict"])
        model.set_similarity_matrix()
        eval_dataset(model, similarity_measure, base_data_loader, args.encoded_test_file, args.load_encoded_test,
                     args.encoded_kb_file, args.load_encoded_kb, intermedia_stuff, args.method, args.trg_encoding_num,
                     args.mid_encoding_num,
                     args.result_file, args.record_recall)


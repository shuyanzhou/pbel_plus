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

torch.manual_seed(0)
random.seed(0)


PP_VEC_SIZE = 22
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Batch(BaseBatch):
    def set_src(self, src_tensor, src_lens, src_perm_idx, src_gold_kb_ids):
        self.src_tensor = src_tensor
        self.src_lens = src_lens
        self.src_perm_idx = src_perm_idx
        self.gold_kb_ids = src_gold_kb_ids
        self.src_flag = True

    def set_trg(self, trg_tensor, trg_lens, trg_perm_idx, trg_kb_ids):
        self.trg_tensor = trg_tensor
        self.trg_lens = trg_lens
        self.trg_perm_idx = trg_perm_idx
        self.trg_kb_ids = trg_kb_ids
        self.trg_flag = True

    def set_mega(self, mega_tensor, mega_lens, mega_perm_idx, mega_trg_kb_ids):
        self.mega_tensor = mega_tensor
        self.mega_lens = mega_lens
        self.mega_perm_idx = mega_perm_idx
        self.mega_trg_kb_ids = mega_trg_kb_ids
        self.mega_flag = True
        self.negative_num = 1

    def set_mid(self, mid_tensor, mid_lens, mid_perm_idx, mid_kb_ids):
        self.mid_tensor = mid_tensor
        self.mid_lens = mid_lens
        self.mid_perm_idx = mid_perm_idx
        self.mid_kb_ids = mid_kb_ids
        self.mid_flag = True

    def to(self, device):
        if self.src_flag:
            self.src_tensor = self.src_tensor.to(device)
            self.src_lens = self.src_lens.to(device)
            self.src_perm_idx = self.src_perm_idx.to(device)
        if self.trg_flag:
            self.trg_tensor = self.trg_tensor.to(device)
            self.trg_lens = self.trg_lens.to(device)
            self.trg_perm_idx = self.trg_perm_idx.to(device)
        if self.mega_flag:
            self.mega_tensor = self.mega_tensor.to(device)
            self.mega_lens = self.mega_lens.to(device)
            self.mega_perm_idx = self.mega_perm_idx.to(device)
        if self.mid_flag:
            self.mid_tensor = self.mid_tensor.to(device)
            self.mid_lens = self.mid_lens.to(device)
            self.mid_perm_idx = self.mid_perm_idx.to(device)

    def get_all(self):
        return  self.src_tensor, self.src_lens, self.src_perm_idx, \
                self.trg_tensor, self.trg_lens, self.trg_perm_idx
    def get_src(self):
        return self.src_tensor, self.src_lens, self.src_perm_idx

    def get_trg(self):
        return self.trg_tensor, self.trg_lens, self.trg_perm_idx

    def get_mid(self):
        return self.mid_tensor, self.mid_lens, self.mid_perm_idx

    def get_mega(self):
        return self.mega_tensor, self.mega_lens, self.mega_perm_idx



class DataLoader(BaseDataLoader):
    def __init__(self, is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab, train_file, dev_file, test_file, trg_encoding_num, mid_encoding_num):
        super(DataLoader,self).__init__(is_train, map_file, batch_size, mega_size, use_panphon, use_mid, share_vocab,
                                        "<UNK>", train_file, dev_file, test_file, trg_encoding_num, mid_encoding_num)

    def new_batch(self):
        return Batch()

    def load_all_data(self, file_name, str_idx, id_idx, x2i_map, encoding_num, type_idx):
        line_tot = 0
        with open(file_name, "r", encoding="utf-8") as fin:
            for line in fin:
                line_tot += 1
                tks = line.strip().split(" ||| ")
                if encoding_num == 1:
                    # make it a list
                    string = [x2i_map[char] for char in tks[str_idx]]
                    string = [string]
                else:
                    all_string = []
                    for i in range(encoding_num):
                        string = [x2i_map[char] for char in ["<" + tks[type_idx] + ">"] +  ["<" + str(i) + ">"] + list(tks[str_idx])]
                        # string = [x2i_map[char] for char in list(tks[str_idx])]
                        all_string.append(string)
                    string = all_string
                yield (string, tks[id_idx])
        print("[INFO] number of lines in {}: {}".format(file_name, str(line_tot)))

    def transform_one_batch(self, batch_data: list) -> list:
        batch_size = len(batch_data)
        batch_lens = torch.LongTensor([len(x) for x in batch_data])
        max_len = torch.max(batch_lens)
        batch_tensor = torch.zeros((batch_size, max_len)).long()
        for idx, (seq, seq_len) in enumerate(zip(batch_data, batch_lens)):
            batch_tensor[idx, :seq_len] = torch.LongTensor(seq)
        # sort
        batch_lens, perm_idx = torch.sort(batch_lens, dim=0, descending=True)
        batch_tensor = batch_tensor[perm_idx]
        # [b, max_len] - > [max_len, b]
        batch_tensor = torch.transpose(batch_tensor, 1, 0)

        # perm idx is used to recover the original order, as src and trg will be different!
        return [batch_tensor, batch_lens, perm_idx]


class LSTMEncoder(Encoder):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, use_panphon, similarity_measure:Similarity,
                 use_mid, share_vocab, mid_vocab_size=0):
        super(LSTMEncoder, self).__init__()
        self.similarity_measure = similarity_measure
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.bilinear = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
        torch.nn.init.xavier_uniform_(self.bilinear, gain=1)
        # self.bilinear = nn.Parameter(torch.eye(self.hidden_size), requires_grad=True)
        self.use_mid = use_mid
        self.share_vocab = share_vocab
        self.mid_vocab_size=mid_vocab_size
        if not use_panphon:
            self.src_lookup = nn.Embedding(src_vocab_size, embed_size)
            self.trg_lookup = nn.Embedding(trg_vocab_size, embed_size)
            torch.nn.init.xavier_uniform_(self.src_lookup.weight, gain=1)
            torch.nn.init.xavier_uniform_(self.trg_lookup.weight, gain=1)
            self.use_panphon = False
        else: # panphon embeddings
            self.pp_linear = nn.Linear(PP_VEC_SIZE, embed_size)
            self.use_panphon = True
            self.src_lookup = None
            self.trg_lookup = None

        self.src_lstm = nn.LSTM(embed_size, int(hidden_size / 2), bidirectional=True)
        self.trg_lstm = nn.LSTM(embed_size, int(hidden_size / 2), bidirectional=True)
        for name, param in self.src_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        for name, param in self.trg_lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

        if use_mid:
            if share_vocab:
                self.mid_lookup = self.src_lookup
                self.mid_lstm = self.src_lstm
            else:
                self.mid_lookup = nn.Embedding(mid_vocab_size, embed_size)
                self.mid_lstm = nn.LSTM(embed_size, int(hidden_size / 2), bidirectional=True)
                torch.nn.init.xavier_uniform_(self.mid_lookup.weight, gain=1)
            # self.bilinear_mid = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size)))
            # torch.nn.init.xavier_uniform_(self.bilinear_mid, gain=1)
            self.bilinear_mid = nn.Parameter(torch.eye(self.hidden_size), requires_grad=True)
        else:
            # self.bilinear_mid = None
            self.bilinear_mid = nn.Parameter(torch.eye(self.hidden_size), requires_grad=True)


    # calc_batch_similarity will return the similarity of the batch
    # while calc encode only return the encoding result of src or trg of the batch
    def calc_encode(self, batch, is_src, is_mega=False, is_mid=False):
        # input: [len, batch] or [len, batch, pp_vec_size]
        # embed: [len, batch, embed_size]
        if is_mid:
            lookup = self.mid_lookup
            lstm = self.mid_lstm
            input, input_lens, perm_idx = batch.get_mid()
        else:
            if is_src:
                lookup = self.src_lookup
                lstm = self.src_lstm
                input, input_lens, perm_idx = batch.get_src()
            else:
                lookup = self.trg_lookup
                lstm = self.trg_lstm
                if is_mega:
                    input, input_lens, perm_idx = batch.get_mega()
                else:
                    input, input_lens, perm_idx = batch.get_trg()

        if not self.use_panphon:
            embeds = lookup(input)
        # no need to lookup, re weight panphon features
        else:
            #[len, batch, pp_vec_size] -> [len, batch, embed_size]
            embeds = self.pp_linear(input)
        packed = pack_padded_sequence(embeds, input_lens, batch_first=False)
        packed_output, (hidden, cached) = lstm(packed)
        # get the last hidden state
        # [2, batch, hidden]
        encoded = hidden
        # [batch, 2, hidden]
        encoded = torch.transpose(encoded, 0, 1).contiguous()
        # combine hidden state of two directions
        # [batch, hidden * 2]
        bi_encoded = encoded.view(-1, self.hidden_size)
        reorder_encoded = bi_encoded[torch.sort(perm_idx, 0)[1]]

        return reorder_encoded

def save_model(model:LSTMEncoder, epoch, loss, optimizer, model_path):
    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "src_vocab_size": model.src_vocab_size,
                "trg_vocab_size": model.trg_vocab_size,
                "mid_vocab_size": model.mid_vocab_size,
                "embed_size": model.embed_size,
                "hidden_size": model.hidden_size,
                "similarity_measure": model.similarity_measure.method,
                "epoch": epoch,
                "loss": loss}, model_path)
    print("[INFO] save model!")

if __name__ == "__main__":
    args = argps()
    if args.is_train:
        data_loader, criterion, similarity_measure = init_train(args, DataLoader)
        model = LSTMEncoder(data_loader.src_vocab_size, data_loader.trg_vocab_size,
                    args.embed_size, args.hidden_size, args.use_panphon,
                    similarity_measure,
                    args.use_mid, args.share_vocab, data_loader.mid_vocab_size)
        optimizer, scheduler = create_optimizer(args.trainer, args.learning_rate, model, args.lr_decay)
        if args.finetune:
            model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
            model.load_state_dict(model["model_state_dict"])
            optimizer.load_state_dict(model["optimizer_state_dict"])
            print("[INFO] load model from epoch {:d} train loss: {:.4f}".format(model_info["epoch"], model_info["loss"]))
        run(data_loader, model, criterion, optimizer, scheduler, similarity_measure, save_model, args)
    else:
        base_data_loader, intermedia_stuff = init_test(args, DataLoader)
        model_info = torch.load(args.model_path + "_" + str(args.test_epoch) + ".tar")
        similarity_measure = Similarity(args.similarity_measure)
        model = LSTMEncoder(model_info["src_vocab_size"], model_info["trg_vocab_size"],
                        args.embed_size, args.hidden_size,
                        use_panphon=args.use_panphon,
                        similarity_measure=similarity_measure,
                        use_mid=args.use_mid,
                        share_vocab=args.share_vocab,
                        mid_vocab_size=model_info.get("mid_vocab_size", 0))

        model.load_state_dict(model_info["model_state_dict"])
        eval_dataset(model, similarity_measure, base_data_loader, args.encoded_test_file, args.load_encoded_test,
                     args.encoded_kb_file, args.load_encoded_kb, intermedia_stuff, args.method, args.trg_encoding_num,
                     args.mid_encoding_num, args.result_file, args.record_recall)

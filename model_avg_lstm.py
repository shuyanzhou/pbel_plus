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
from utils.constant import  RANDOM_SEED, DEVICE, PP_VEC_SIZE
from model_lstm import Batch, DataLoader, LSTMEncoder, save_model

random_seed = RANDOM_SEED
torch.manual_seed(random_seed)
random.seed(random_seed)
device = DEVICE


class AvgLSTMEncoder(LSTMEncoder):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_size, hidden_size, use_panphon, similarity_measure:Similarity,
                 use_mid, share_vocab, mid_vocab_size=0):
        super(AvgLSTMEncoder, self).__init__(src_vocab_size, trg_vocab_size, embed_size, hidden_size, use_panphon, similarity_measure,
                 use_mid, share_vocab, mid_vocab_size)


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
        output, _ = pad_packed_sequence(packed_output)

        # [batch, len, 2 * hidden]
        encoded = torch.transpose(output, 0, 1)
        # [batch, 2 * hidden]
        avg_encoded = torch.sum(encoded, dim=1) / input_lens.unsqueeze(-1).float()
        reorder_encoded = avg_encoded[torch.sort(perm_idx, 0)[1]]

        return reorder_encoded


if __name__ == "__main__":
    args = argps()
    if args.is_train:
        data_loader, criterion, similarity_measure = init_train(args, DataLoader)
        model = AvgLSTMEncoder(data_loader.src_vocab_size, data_loader.trg_vocab_size,
                    args.embed_size, args.hidden_size, args.use_panphon,
                    similarity_measure,
                    args.use_mid, args.share_vocab, data_loader.mid_vocab_size)
        optimizer, scheduler = create_optimizer(args.trainer, args.learning_rate, model, args.lr_decay)
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
        model = AvgLSTMEncoder(model_info["src_vocab_size"], model_info["trg_vocab_size"],
                        args.embed_size, args.hidden_size,
                        use_panphon=args.use_panphon,
                        similarity_measure=similarity_measure,
                        use_mid=args.use_mid,
                        share_vocab=args.share_vocab,
                        mid_vocab_size=model_info.get("mid_vocab_size", 0))

        model.load_state_dict(model_info["model_state_dict"], strict=False)
        model.set_similarity_matrix()
        eval_dataset(model, similarity_measure, base_data_loader, args.encoded_test_file, args.load_encoded_test,
                     args.encoded_kb_file, args.load_encoded_kb, intermedia_stuff, args.method, args.trg_encoding_num,
                     args.mid_encoding_num, args.result_file, args.record_recall)

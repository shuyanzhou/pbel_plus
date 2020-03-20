from collections import defaultdict
import functools
import torch
import random
import sys
import numpy as np
import time
from models.base_train import BaseDataLoader, Encoder, FileInfo, list2nparr, append_multiple_encodings, merge_encodings
from utils.similarity_calculator import Similarity
from typing import List
from utils.constant import RANDOM_SEED, PP_VEC_SIZE, DEVICE

device = DEVICE
print = functools.partial(print, flush=True)
# load data for ONE side (e.g KB or test data)


def get_rank(scores:np.ndarray, kb_id:np.ndarray, kb_entity_string:list, topk=100):
    limit = min(len(scores), topk)
    # find the index of top_limit elements
    max_idx = np.argpartition(scores, -limit)[-limit:]
    # sort these index by their scores
    ranked_idxs = max_idx[np.argsort(scores[max_idx])][::-1]
    ranked_ids = kb_id[ranked_idxs]
    ranked_entity_string = [kb_entity_string[i] for i in ranked_idxs]
    ranked_scores = scores[ranked_idxs]

    return ranked_ids, ranked_entity_string, ranked_scores

def update_recall(gold_id:int, top_predict_ids:np.ndarray, recall_dict:dict, topk_list:list):
    for topk in topk_list:
        if gold_id in top_predict_ids[:topk]:
            recall_dict[str(topk)] += 1

def record_result(opened_file, opened_file_string, plain_text, top_predict_ids:np.ndarray, top_predict_strings: list, top_predict_scores:np.ndarray):
    opened_file.write(plain_text + " ||| ")
    id_score_pair = [str(x) + " | " + str(y) for x, y in zip(top_predict_ids, top_predict_scores)]
    id_score_pair = " || ".join(id_score_pair)
    opened_file.write(id_score_pair + "\n")

    opened_file_string.write(plain_text + " ||| ")
    string_score_pair = [str(x) + " | " + str(y) for x, y in zip(top_predict_strings, top_predict_scores)]
    string_score_pair = " || ".join(string_score_pair)
    opened_file_string.write(string_score_pair + "\n")

def calc_scores(scores, data_plain, gold_kb_ids, kb_ids, kb_entity_string, result_file: list, record_recall, recall_file, topk_list):
    print("[INFO] current score matrix shape: ", str(scores.shape))
    assert scores.shape[0] == len(data_plain) and len(data_plain) == len(gold_kb_ids)
    for idx, (cur_scores, plain_text, gold_kb_id) in enumerate(zip(scores, data_plain, gold_kb_ids)):
        ranked_ids, ranked_entity_string, ranked_scores = get_rank(cur_scores, kb_ids, kb_entity_string)
        record_result(result_file[0], result_file[1], plain_text, ranked_ids, ranked_entity_string, ranked_scores)
        if record_recall:
            update_recall(gold_kb_id, ranked_ids, recall_file, topk_list)

def close_file_list(file_list):
    for f in file_list:
        f.close()


def exact_match(original_scores, test_data_plain, kb_entity_strings):
    i, j = original_scores.shape
    assert len(test_data_plain) == i
    assert len(kb_entity_strings) == j
    updated_scores = np.copy(original_scores)
    for idx, data_plain in enumerate(test_data_plain):
        if data_plain in kb_entity_strings:
            target_idx = kb_entity_strings.index(data_plain)
            updated_scores[idx, target_idx] = 1000.0
    return updated_scores



def calc_result(test_data_encodings:np.ndarray, test_gold_kb_ids:np.ndarray, test_data_plain:list,
                kb_encodings:np.ndarray, kb_ids:np.ndarray, kb_entity_string:list,
                intermediate_info:dict,
                method, similarity_calculator: Similarity,
                save_files:dict, trg_encoding_num, mid_encoding_num, topk_list = (1, 2, 5, 10, 30),
                record_recall=False, use_exact_match=True):
    pieces=100
    # no pivoting, base method
    tot = float(test_data_encodings.shape[0])
    # base method
    base_recall = {str(topk):0 for topk in topk_list}
    base_result_file = open(save_files["no_pivot"], "w+", encoding="utf-8")
    base_result_string_file = open(save_files["no_pivot_str"], "w+", encoding="utf-8")
    base_files = [base_result_file, base_result_string_file]

    # split_kb_encodings = np.split(kb_encodings, trg_encoding_num, axis=0)
    # kb_size = split_kb_encodings[0].shape[0]
    # base_scores = np.zeros((tot, kb_size)) - 10000
    # st_time = time.time()
    # for cur_kb_encodings in split_kb_encodings:
    base_scores = similarity_calculator(test_data_encodings, kb_encodings,
                                        is_src_trg=True, split=True, pieces=pieces, negative_sample=None, encoding_num=trg_encoding_num)
    # calc exact match
    if use_exact_match:
        base_scores = exact_match(base_scores, test_data_plain, kb_entity_string)
    calc_scores(base_scores, test_data_plain, test_gold_kb_ids, kb_ids, kb_entity_string, base_files, record_recall, base_recall, topk_list)

    print("===============encoding recall===============")
    for topk, recall in base_recall.items():
        print("[INFO] top {}: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))

    close_file_list(base_files)

    if method == "pivoting":
        pivot_recall = {str(topk):0 for topk in topk_list}
        pivot_encodings = intermediate_info["encodings"]["pivot"]
        pivot_kb_ids = intermediate_info["kb_id"]["pivot"]
        pivot_kb_ids = np.concatenate([kb_ids, pivot_kb_ids])

        pivot_kb_entity_string = intermediate_info["plain_text"]["pivot"]
        pivot_kb_entity_string = kb_entity_string + pivot_kb_entity_string

        pivot_result_file = open(save_files["pivot"], "w+", encoding="utf-8")
        pivot_result_string_file = open(save_files["pivot_str"], "w+", encoding="utf-8")
        pivot_files = [pivot_result_file, pivot_result_string_file]

        pivot_scores = similarity_calculator(test_data_encodings, pivot_encodings,
                                             is_src_trg=False, split=True, pieces=pieces, negative_sample=None, encoding_num=mid_encoding_num)
        combined_scores = np.hstack([base_scores, pivot_scores])
        # exact match
        if use_exact_match:
            combined_scores = exact_match(combined_scores, test_data_plain, pivot_kb_entity_string)
        calc_scores(combined_scores, test_data_plain, test_gold_kb_ids, pivot_kb_ids, pivot_kb_entity_string, pivot_files, record_recall, pivot_recall, topk_list)

        print("===============pivoting recall===============")
        for topk, recall in pivot_recall.items():
            print("[INFO] top {}: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))

        close_file_list(pivot_files)

def get_kb_id(fname, str_idx, id_idx):
    gold_kb_id = []
    plain_text = []
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            gold_kb_id.append(int(tks[id_idx]))
            plain_text.append(tks[str_idx])
    gold_kb_id = np.array(gold_kb_id)
    return gold_kb_id, plain_text

def get_encodings(model: Encoder, data_loader: BaseDataLoader, load_encoding: bool, save_file, is_src, is_mid, encoding_num):
    if not load_encoding:
        batches = data_loader.create_batches("test", is_src=is_src, is_mid=is_mid)
        # encodings = np.empty((0, encoder.hidden_size*2))
        encodings = [[] for _ in range(encoding_num)]
        start_time = time.time()
        for idx, batch in enumerate(batches):
            if (idx + 1) % 10000 == 0:
                print("[INFO] process {} batches, using {:.2f} seconds".format(idx + 1, time.time() - start_time))
            cur_encodings = np.array(model.calc_encode(batch, is_src=is_src, is_mid=is_mid).cpu())
            append_multiple_encodings(encodings, cur_encodings, encoding_num)
        encodings = list2nparr(encodings, model.hidden_size, merge=True)
        print("[INFO] encoding shape: {}".format(str(encodings.shape)))
        print("[INFO] done all {} batches, using {:.2f} seconds".format(len(batches), time.time() - start_time))
        # np.save(save_file, encodings)
        # print("[INFO] save test encodings!")
    else:
        encodings = np.load(save_file)
        print("[INFO] load test encodings!")
        print(encodings.shape)

    if is_mid:
        kb_ids, data_plain = get_kb_id(data_loader.test_file.mid_file_name,
                                       data_loader.test_file.mid_str_idx,
                                       data_loader.test_file.mid_id_idx)
    else:
        if is_src:
            kb_ids, data_plain = get_kb_id(data_loader.test_file.src_file_name,
                                           data_loader.test_file.src_str_idx,
                                           data_loader.test_file.src_id_idx)
        else:
            kb_ids, data_plain = get_kb_id(data_loader.test_file.trg_file_name,
                                           data_loader.test_file.trg_str_idx,
                                           data_loader.test_file.trg_id_idx)

    assert kb_ids.shape[0] == int(encodings.shape[0] / encoding_num) \
           and len(data_plain) == int(encodings.shape[0] / encoding_num), \
        (kb_ids.shape[0], int(encodings.shape[0] / encoding_num), len(data_plain))

    return encodings, kb_ids, data_plain


# intermediate_stuff contains arguments from pivoting et al
# method, pivoting et al
def eval_dataset(model:Encoder, similarity_calculator: Similarity,
                 base_data_loader:BaseDataLoader,
                 encoded_test_file, load_encoded_test,
                 encoded_kb_file, load_encoded_kb,
                 intermediate_stuff,
                 method,
                 trg_encoding_num,
                 mid_encoding_num,
                 result_files:dict,
                 record_recall: bool):
    with torch.no_grad():
        model.eval()
        model.to(device)
        encoded_test, test_gold_kb_id, test_data_plain = get_encodings(model, base_data_loader, load_encoded_test, encoded_test_file, is_src=True, is_mid=False, encoding_num=1)
        encoded_kb, kb_ids, kb_entity_string = get_encodings(model, base_data_loader, load_encoded_kb, encoded_kb_file, is_src=False, is_mid=False, encoding_num=trg_encoding_num)
        intermediate_info = {}
        if method != "base":
            intermediate_encodings = {}
            intermediate_kb_id = {}
            intermediate_plain_text = {}
            for stuff in intermediate_stuff:
                # name is used to present the contain of this intermediate stuff
                name, data_loader, encoded_file, load_encoded, is_src, is_mid = stuff
                encoded_stuff, gold_kb_id, plain_text = get_encodings(model, data_loader, load_encoded, encoded_file, is_src=is_src, is_mid=is_mid, encoding_num=mid_encoding_num)
                intermediate_encodings[name] = encoded_stuff
                intermediate_kb_id[name] = gold_kb_id
                intermediate_plain_text[name] = plain_text
            intermediate_info["encodings"] = intermediate_encodings
            intermediate_info["kb_id"] = intermediate_kb_id
            intermediate_info["plain_text"] = intermediate_plain_text
        start_time = time.time()
        calc_result(encoded_test, test_gold_kb_id, test_data_plain,
                    encoded_kb, kb_ids, kb_entity_string,
                    intermediate_info, method, similarity_calculator, result_files, trg_encoding_num, mid_encoding_num, record_recall=record_recall)

        print("[INFO] take {:.4f}s to calculate similarity".format(time.time() - start_time))

# reset the pad embedding to 0 at test time
def reset_unk_weight(model):
    for name, param in model.state_dict().items():
        if "src_lookup" in name or "trg_lookup" in name or "mid_lookup" in name:
            embed_size = param.shape[1]
            param[0] = torch.zeros((1, embed_size))

def init_test(args, DataLoader):
    test_file = FileInfo()
    test_file.set_src(args.test_file, args.test_str_idx, args.test_id_idx)
    test_file.set_trg(args.kb_file, args.kb_str_idx, args.kb_id_idx, args.kb_type_idx)
    base_data_loader = DataLoader(is_train=False, args=args, train_file=None, dev_file=None, test_file=test_file)
    intermediate_stuff = []
    if args.method != "base":
        for stuff in args.intermediate_stuff:
            name, file_name, str_idx, id_idx, type_idx, encoded_file, load_encoded, is_src, is_mid = stuff
            inter_file = FileInfo()
            if is_mid:
                inter_file.set_mid(file_name, str_idx, id_idx, type_idx)
            else:
                if is_src:
                    inter_file.set_src(file_name, str_idx, id_idx)
                else:
                    inter_file.set_trg(file_name, str_idx, id_idx, type_idx)
            inter_data_loader = DataLoader(is_train=False, args=args, train_file=None, dev_file=None, test_file=inter_file)
            intermediate_stuff.append((name, inter_data_loader, encoded_file, load_encoded, is_src, is_mid))

    return base_data_loader, intermediate_stuff



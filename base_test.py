from collections import defaultdict
import functools
import torch
import random
import sys
sys.path.append("/home/shuyanzh/workshop/cmu_lorelei_edl/")
import numpy as np
import time
from mention_matching.pbel.base_train import BaseDataLoader, Encoder, FileInfo
from mention_matching.pbel.similarity_calculator import Similarity

PP_VEC_SIZE = 22
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print = functools.partial(print, flush=True)
# load data for ONE side (e.g KB or test data)


def list2nparr(org_list:list, hidden_size:int):
    encodings = np.array(org_list[:-1])
    encodings = np.reshape(encodings, (-1, hidden_size))
    # last batch might not match the size
    encodings = np.append(encodings, np.array(org_list[-1]), axis=0)
    return encodings

def get_rank(scores:np.ndarray, kb_id:np.ndarray, kb_entity_string:list, topk=30):
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

def calc_result(test_data_encodings:np.ndarray, test_gold_kb_ids:np.ndarray, test_data_plain:list,
                kb_encodings:np.ndarray, kb_ids:np.ndarray, kb_entity_string:list,
                intermediate_info:dict,
                method, similarity_calculator: Similarity, bilinear_tensor: torch.Tensor, save_files:dict, topk_list = (1, 2, 5, 10, 30),
                record_recall=False):
    pieces=500
    # no pivoting, base method
    tot = float(test_data_encodings.shape[0])
    # base method
    base_recall = {str(topk):0 for topk in topk_list}
    base_result_file = open(save_files["no_pivot"], "w+", encoding="utf-8")
    base_result_string_file = open(save_files["no_pivot_str"], "w+", encoding="utf-8")
    base_files = [base_result_file, base_result_string_file]
    base_scores = similarity_calculator(test_data_encodings, kb_encodings, bilinear_tensor, split=True, pieces=pieces, negative_sample=None)
    calc_scores(base_scores, test_data_plain, test_gold_kb_ids, kb_ids, kb_entity_string, base_files, record_recall, base_recall, topk_list)

    for topk, recall in base_recall.items():
        print("[INFO] top {} recall: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))

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

        pivot_scores = similarity_calculator(test_data_encodings, pivot_encodings, bilinear_tensor, split=True, pieces=pieces, negative_sample=None)
        combined_scores = np.hstack([base_scores, pivot_scores])
        calc_scores(combined_scores, test_data_plain, test_gold_kb_ids, pivot_kb_ids, pivot_kb_entity_string, pivot_files, record_recall, pivot_recall, topk_list)

        print("===============pivoting recall===============")
        for topk, recall in pivot_recall.items():
            print("[INFO] top {} recall: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))

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

def get_encodings(model: Encoder, data_loader: BaseDataLoader, load_encoding: bool, save_file, is_src):
    if not load_encoding:
        batches = data_loader.create_batches("test", is_src)
        # encodings = np.empty((0, encoder.hidden_size*2))
        encodings = []
        start_time = time.time()
        for idx, batch in enumerate(batches):
            if (idx + 1) % 10000 == 0:
                print("[INFO] process {} batches, using {:.2f} seconds".format(idx + 1, time.time() - start_time))
            encodings.append(np.array(model.calc_encode(batch, is_src).cpu()))

        # encoding = [batch_num, batch_size, hidden_state * 2]
        encodings = list2nparr(encodings, model.hidden_size)
        print(encodings.shape)
        print("[INFO] done all {} batches, using {:.2f} seconds".format(len(batches), time.time() - start_time))
        # np.save(save_file, encodings)
        # print("[INFO] save test encodings!")
    else:
        encodings = np.load(save_file)
        print("[INFO] load test encodings!")
        print(encodings.shape)

    if is_src:
        kb_ids, data_plain = get_kb_id(data_loader.test_file.src_file_name,
                                       data_loader.test_file.src_str_idx,
                                       data_loader.test_file.src_id_idx)
    else:
        kb_ids, data_plain = get_kb_id(data_loader.test_file.trg_file_name,
                                       data_loader.test_file.trg_str_idx,
                                       data_loader.test_file.trg_id_idx)

    assert kb_ids.shape[0] == encodings.shape[0] \
           and len(data_plain) == encodings.shape[0], \
        (kb_ids.shape[0], encodings.shape[0], len(data_plain))

    return encodings, kb_ids, data_plain


# intermediate_stuff contains arguments from pivoting et al
# method, pivoting et al
def eval_dataset(model:Encoder, similarity_calculator: Similarity,
                 base_data_loader:BaseDataLoader,
                 encoded_test_file, load_encoded_test,
                 encoded_kb_file, load_encoded_kb,
                 intermediate_stuff,
                 method, result_files:dict,
                 record_recall: bool):
    with torch.no_grad():
        model.eval()
        model.to(device)
        encoded_test, test_gold_kb_id, test_data_plain = get_encodings(model, base_data_loader, load_encoded_test, encoded_test_file, is_src=True)
        encoded_kb, kb_ids, kb_entity_string = get_encodings(model, base_data_loader, load_encoded_kb, encoded_kb_file, is_src=False)
        intermediate_info = {}
        if method != "base":
            intermediate_encodings = {}
            intermediate_kb_id = {}
            intermediate_plain_text = {}
            for stuff in intermediate_stuff:
                # name is used to present the contain of this intermediate stuff
                name, data_loader, encoded_file, load_encoded, is_src = stuff
                encoded_stuff, gold_kb_id, plain_text = get_encodings(model, data_loader, load_encoded, encoded_file, is_src=is_src)
                intermediate_encodings[name] = encoded_stuff
                intermediate_kb_id[name] = gold_kb_id
                intermediate_plain_text[name] = plain_text
            intermediate_info["encodings"] = intermediate_encodings
            intermediate_info["kb_id"] = intermediate_kb_id
            intermediate_info["plain_text"] = intermediate_plain_text
        start_time = time.time()
        calc_result(encoded_test, test_gold_kb_id, test_data_plain,
                    encoded_kb, kb_ids, kb_entity_string,
                    intermediate_info, method, similarity_calculator, model.bilinear, result_files, record_recall=record_recall)

        print("[INFO] take {:.4f}s to calculate similarity".format(time.time() - start_time))


def init_test(args, DataLoader):
    test_file = FileInfo()
    test_file.set_src(args.test_file, args.test_str_idx, args.test_id_idx)
    test_file.set_trg(args.kb_file, args.kb_str_idx, args.kb_id_idx)
    base_data_loader = DataLoader(False, args.map_file, args.batch_size, args.mega_size, args.use_panphon, test_file=test_file)
    intermediate_stuff = []
    if args.method != "base":
        for stuff in args.intermediate_stuff:
            name, file_name, str_idx, id_idx, encoded_file, load_encoded, is_src = stuff
            inter_file = FileInfo()
            if is_src:
                inter_file.set_src(file_name, str_idx, id_idx)
            else:
                inter_file.set_trg(file_name, str_idx, id_idx)
            inter_data_loader = DataLoader(False, args.map_file, args.batch_size, args.mega_size, args.use_panphon,
                                           test_file=inter_file)
            intermediate_stuff.append((name, inter_data_loader, encoded_file, load_encoded, is_src))

    return base_data_loader, intermediate_stuff



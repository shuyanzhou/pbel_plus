import os
import sys
import epitran
from collections import defaultdict, Counter
import numpy as np

# get KB type map
def get_type_map():
    kb_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
    entity_type_map = {}
    id_name_map = {}
    with open(kb_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            id, name, type = tks
            entity_type_map[name]=type
            id_name_map[id] = name
    return entity_type_map, id_name_map

def calc_dedup_recall(fname):
    tot = 0
    acc = 0
    all_mentions = set()
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            gold, mention, candidate = tks[1], tks[2], tks[4]
            if mention not in all_mentions:
                if gold in candidate.split(" || "):
                    acc += 1
                all_mentions.add(mention)
    tot = len(all_mentions)
    print(acc, tot, acc/tot)

# get analysis file
def get_analysis_file(lang, gold_file, result_file, new_result_file):
    epitran_map = {"hi": "hin-Deva",
                   "am": "amh-Ethi",
                   "th": "tha-Thai",
                   "tr": "tur-Latn",
                   "ta": "tam-Taml",
                   "id": "ind-Latn",
                   "mr": "mar-Deva",
                   "en": "eng-Latn",
                   "ti": "tir-Ethi",
                   "te": "tel-Telu",
                   "lo": "lao-Laoo",
                   "om": "orm-Latn",
                   "kw": "kin-Latn",
                   "si": "sin-Sinh",
                   "il10":"sin-Sinh",
                   "il5":"tir-Ethi",
                   "il6":"som-Latn",
                   "il9":"run-Latn"}

    epi = epitran.Epitran(epitran_map[lang])
    entity_type_map, id_name_map = get_type_map()

    tot = 0
    recall_dict = {'1': 0, '2': 0, '5': 0, '10': 0, '20': 0, '30': 0}

    if not os.path.exists(new_result_file):
        with open(gold_file, "r", encoding="utf-8") as fg:
            with open(result_file, "r", encoding="utf-8") as fr:
                with open(new_result_file, "w+", encoding="utf-8") as fout:
                    for gold_line, result_line in zip(fg, fr):
                        tot += 1
                        gold_en_string, mention = gold_line.strip().split(" ||| ")[1:3]
                        gold_type = entity_type_map.get(gold_en_string, "UNK")
                        _, entity_info = result_line.split(" ||| ")
                        mention_ipa = epi.transliterate(mention)
                        entity_info = [x.split(" | ") for x in entity_info.split(" || ")]
                        entity_ids = [x[0] for x in entity_info]
                        entity_scores = [float(x[1]) for x in entity_info]
                        # remove duplicate
                        entity_max_score = defaultdict(float)
                        for eid, escore in zip(entity_ids, entity_scores):
                            entity_max_score[eid] = max(entity_max_score[eid], escore)
                        entity_ids = [*entity_max_score]
                        scores = [entity_max_score[x] for x in entity_ids]
                        sort_idx = np.argsort(np.array(scores))[::-1]
                        sort_idx = sort_idx[:30]
                        entity_ids = [entity_ids[x] for x in sort_idx]
                        entity_strings = [id_name_map[x] for x in entity_ids]
                        fout.write(gold_type + " ||| " + gold_en_string + " ||| " + mention_ipa + " ||| " + mention + " ||| " + " || ".join(entity_strings) + "\n")

                        for topk in recall_dict.keys():
                            topk = int(topk)
                            if gold_en_string in entity_strings[:topk]:
                                recall_dict[str(topk)] += 1

        for topk, recall in recall_dict.items():
            print("[INFO] top {}: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))

    print("[INFO] unique result")
    calc_dedup_recall(new_result_file)


def get_diff(f1, f2, fdiff):
    base = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/analysis"
    f1 = os.path.join(base, f1)
    f2 = os.path.join(base, f2)
    fdiff = os.path.join(base, fdiff)
    with open(f1, "r", encoding="utf-8") as fin1:
        with open(f2, "r", encoding="utf-8") as fin2:
            with open(fdiff, "w+", encoding="utf-8") as fout:
                for l1, l2 in zip(fin1, fin2):
                    tks = l1.strip().split(" ||| ")
                    answer, entity = tks[1], tks[-1].split(" || ")
                    tks = l2.strip().split(" ||| ")
                    answer2, entity2 = tks[1], tks[-1].split(" || ")
                    assert answer == answer2
                    if answer not in entity and answer in entity2:
                        fout.write(l1)
                        fout.write("\n")
                        fout.write(l2)
                        fout.write("=======================\n")



def bucket_result(fname, criterion):
    acc_counter = Counter()
    tot_counter = Counter()
    if criterion == "length":
        bucket = [(-1000, -11), (-11, -7), (-7, -3), (-3, 2), (2, 6), (6, 10), (10, 14), (14, 18), (18, 22), (22, 1000)]
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                tks = line.strip().split(" ||| ")
                answer, mention, entity = tks[1], tks[3], tks[4].split(" || ")
                if tks[0] != "ORG":
                    continue
                length_diff = len(answer) - len(mention)
                diff_bucket = [b for b in bucket if length_diff > b[0] and length_diff <= b[1]][0]
                tot_counter[diff_bucket] += 1
                if answer in entity:
                    acc_counter[diff_bucket] += 1
        print("answer length - mention length:")
        for b in bucket:
            if tot_counter[b] != 0:
                print(b, acc_counter[b], tot_counter[b], float(acc_counter[b]) / float(tot_counter[b]))
            
    elif criterion == "type":
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                tks = line.strip().split(" ||| ")
                answer_type, answer, mention, entity = tks[0], tks[1], tks[3], tks[4].split(" || ")
                tot_counter[answer_type] += 1
                if answer in entity:
                    acc_counter[answer_type] += 1

        for k, tot in tot_counter.items():
            print(k, float(acc_counter[k]), tot, float(acc_counter[k]) / tot)

def get_file_name(model, data, lang, pivot):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/"
    if data == "me":
        gold_file = os.path.join(base_path, "data", "me_test_en-{}_links".format(lang))
        result_file = os.path.join(base_path, "results", "me{}en-{}_{}.id".format(pivot, lang, model))
        new_result_file = os.path.join(base_path, "results", "analysis",
                                       "me{}en-{}_{}.anl".format(pivot, lang, model))
    elif data == "ee":
        gold_file = os.path.join(base_path, "data", "ee_test_en-{}_links".format(lang))
        result_file = os.path.join(base_path, "results", "ee{}en-{}_{}.id".format(pivot, lang, model))
        new_result_file = os.path.join(base_path, "results", "analysis",
                                       "ee{}en-{}_{}.anl".format(pivot, lang, model))
    else:
        raise NotImplementedError

    return gold_file, result_file, new_result_file


if __name__ == "__main__":
    lang = sys.argv[1]
    test_data = sys.argv[2]
    pivot = sys.argv[3]
    model = sys.argv[4]
    model2 = sys.argv[5] if len(sys.argv) >= 6 else None

    gold_file, result_file, new_result_file = get_file_name(model, test_data, lang, pivot)
    get_analysis_file(lang, gold_file, result_file, new_result_file)

    if model2 is not None:
        gold_file, result_file, new_result_file = get_file_name(model2, test_data, lang, pivot)
        get_analysis_file(lang, gold_file, result_file, new_result_file)

    # get_diff("unique_mend_ee_{}en-{}_{}.anl".format(PIVOT, lang, model),
    #          "unique_mend_ee_{}en-{}_{}.anl".format(PIVOT, lang, model2),
    #          "unique_mend_ee_{}en-{}_{}-{}.diff".format(PIVOT, lang, model, model2))
    # base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/"

    # result_file = os.path.join(base_path, "results", "analysis", "unique_mend_ee_en-{}_{}.anl".format(lang, model))
    # bucket_result(result_file, "length")
    # bucket_result(result_file, "type")


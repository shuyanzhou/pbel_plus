import os
import numpy as np
from collections import defaultdict
import epitran
import sys

n=20

def merge(suffix, base_path, file_name):
    mention_entity_map = defaultdict(lambda:defaultdict(float))
    mention_list = []
    for i in range(n):
        split_file = os.path.join(base_path, file_name + "_" + str(i) + "." + suffix)
        with open(split_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                tks = line.strip().split(" ||| ")
                mention, entity_info = tks[0], tks[1].split(" || ")
                entity = [x.split(" | ")[0] for x in entity_info]
                entity_score = [x.split(" | ")[1] for x in entity_info]
                for e, es in zip(entity, entity_score):
                    mention_entity_map[str(line_idx) + "_" + mention][e] = max(mention_entity_map[str(line_idx) + "_" + mention][e], float(es))
                if i == 0:
                    mention_list.append(str(line_idx) + "_" + mention)
    assert len(mention_list)  == len(mention_entity_map)
    print("there are {:d} queries".format(len(mention_list)))

    save_path = os.path.join(base_path, "../", file_name + "." + suffix)
    with open(save_path, "w+", encoding="utf-8") as f:
        for mention in mention_list:
            entity_info = mention_entity_map[mention]
            mention = mention.split("_")[1:]
            mention = "_".join(mention)
            entity = [*entity_info]
            entity_score = np.array([entity_info[x] for x in entity])
            sort_idx = np.argsort(entity_score)[::-1]
            sort_idx = sort_idx[:30]
            entity = [entity[x] for x in sort_idx]
            entity_score = entity_score[sort_idx]
            entity_info = [" | ".join([str(x), str(y)]) for x, y in zip(entity, entity_score)]
            f.write("{} ||| {}\n".format(mention, " || ".join(entity_info)))

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



def get_analysis_file(lang, model, gold_file, result_file, new_result_file):
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
                   "il5":"tir-Ethi"}

    epi = epitran.Epitran(epitran_map[lang])
    entity_type_map, id_name_map = get_type_map()

    tot = 0
    recall_dict = {'1': 0, '2': 0, '5': 0, '10': 0, '20': 0, '30': 0}
    with open(gold_file, "r", encoding="utf-8") as fg:
        with open(result_file, "r", encoding="utf-8") as fr:
            with open(new_result_file, "w+", encoding="utf-8") as fout:
                for gold_line, result_line in zip(fg, fr):
                    tot += 1
                    gold_en_string, mention = gold_line.strip().split(" ||| ")[1:3]
                    gold_type = entity_type_map.get(gold_en_string, "NAN")
                    _, entity_info = result_line.split(" ||| ")
                    mention_ipa = epi.transliterate(mention)
                    entity_info = [x.split(" | ") for x in entity_info.split(" || ")]
                    entity_ids = [x[0] for x in entity_info]
                    entity_strings = [id_name_map[x] for x in entity_ids]
                    # find entity strings
                    score = [x[1] for x in entity_info]
                    fout.write(gold_type + " ||| " + gold_en_string + " ||| " + mention_ipa + " ||| " + mention + " ||| " + " || ".join(entity_strings) + "\n")

                    for topk in recall_dict.keys():
                        topk = int(topk)
                        if gold_en_string in entity_strings[:topk]:
                            recall_dict[str(topk)] += 1

    for topk, recall in recall_dict.items():
        print("[INFO] top {}: {:.2f}/{:.2f}={:.4f}".format(topk, recall, tot, recall / tot))


if __name__ == "__main__":
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/split"


    PIVOT = sys.argv[1]
    lang = sys.argv[2]
    model = sys.argv[3]
    test_data = sys.argv[4]
    # PIVOT = "pivot_"
    # lang = "mr"
    # model = "en-hi_ee_cosine-hinge_grapheme_aka"
    # test_data = "ee_mend"
    # test_data = "ee"

    if test_data == "me":
        spl_result_file = "me_test{}en-{}_{}".format(PIVOT, lang, model)
        data_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/"
        gold_file = os.path.join(data_path, "data", "me_en-{}_links".format(lang))
        result_file = os.path.join(data_path, "results", "me{}en-{}_{}.id".format(PIVOT, lang, model))
        new_result_file = os.path.join(data_path, "results", "analysis", "me{}en-{}_{}.anl".format(PIVOT, lang, model))

    elif test_data == "ee":
        spl_result_file = "ee_test{}en-{}_{}".format(PIVOT, lang, model)
        data_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/"
        gold_file = os.path.join(data_path, "data", "ee_test_en-{}_links".format(lang))
        result_file = os.path.join(data_path, "results", "ee_test{}en-{}_{}.id".format(PIVOT, lang, model))
        new_result_file = os.path.join(data_path, "results", "analysis", "ee_test{}en-{}_{}.anl".format(PIVOT, lang, model))
    else:
        sys.exit(0)

    merge("id", base_path, spl_result_file)
    merge("str", base_path, spl_result_file)
    get_analysis_file(lang, model, gold_file, result_file, new_result_file)
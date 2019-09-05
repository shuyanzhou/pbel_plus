import os
import sys
from collections import defaultdict
import numpy as np
def load_result(fname, save_file, mention_list, beta):
    with open(fname, "r", encoding="utf-8") as fin:
        candidate_map = defaultdict(list)
        for idx, line in enumerate(fin):
            tks = line.strip().split(" ||| ")
            mention = mention_list[idx]
            candidate = tks[1].split(" || ")
            candid = [x.split(" | ")[0] for x in candidate][:30]
            score = np.array([float(x.split(" | ")[1]) for x in candidate])
            score[score > 1] = 1
            score = score *  beta
            score = np.exp(score) / np.sum(np.exp(score))
            score = list(score)[:30]
            merge_info = [f"{x} | {y}" for x, y in zip(candid, score)]
            merge_info = " || ".join(merge_info)
            candidate_map[mention] = [candid, score, merge_info]


    with open(save_file + ".map", "w+", encoding="utf-8") as fo1, open(save_file + "_meprior", "w+", encoding="utf-8") as fo2:
        for mention, (candid, score, merge_info) in candidate_map.items():
            fo1.write("{} ||| {}\n".format(mention, " || ".join(candid)))
            fo2.write("{} ||| {}\n".format(mention, merge_info))



def extract_mention(fname):
    mention_list = []
    answer_list = []
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            tks = line.strip().split(" ||| ")
            mention_list.append(tks[2])
            answer_list.append(tks[0])
    return mention_list, answer_list

def get_candid(fname):
    with open(fname, "r", encoding="utf-8") as fin:
        candidate_map = defaultdict(list)
        for idx, line in enumerate(fin):
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            mention = tks[0]
            candidate = tks[1].split(" || ")
            candid = [x.split(" | ")[0] for x in candidate]
            score = [float(x.split(" | ")[1]) for x in candidate]
            candidate_map[mention] = [candid, score]

    return candidate_map


def calc_recall(query_file, candid_file):
    topn = 30
    candid_map = get_candid(candid_file)
    tot_log_score = []
    with open(query_file, "r", encoding="utf-8") as fin:
        tot = 0
        recall = 0
        top1 = 0
        for line in fin:
            tks = line.strip().split(" ||| ")
            answer, mention = tks[0], tks[2]
            candid = candid_map.get(mention, [[], []])[0]
            candid_score = candid_map.get(mention, [[], []])[1]
            candid = candid[: min(topn, len(candid))]
            candid_score = candid_score[: min(topn, len(candid_score))]

            if answer in candid:
                recall += 1
                if answer in candid[:1]:
                    top1 += 1
                tot_log_score.append(candid_score[candid.index(answer)])
            tot += 1

    tot_log_score = np.array(tot_log_score)
    log_score = np.sum(np.log(tot_log_score))

    print(recall, tot, recall / tot, top1, top1 / tot, log_score)

def get_score(candid_info, candid):
    if candid in candid_info[0]:
        score = candid_info[1][candid_info[0].index(candid)]
    else:
        score = 0
    return float(score)

def merge_candidates(wiki_file, app_file, mention_list, save_file, alpha):
    wiki_candid = get_candid(wiki_file)
    for v in wiki_candid.values():
        assert len(v) == 2
        assert len(v[0]) < 20
    app_candid = get_candid(app_file)

    with open(save_file + ".map", "w+", encoding="utf-8") as fo1, open(save_file + "_meprior", "w+", encoding="utf-8") as fo2 :
        for m in mention_list:
            wc = wiki_candid.get(m, [[], []])
            ac = app_candid.get(m, [[], []])

            # if len(wc[0]) != 0:
            #     all_candid = wc[0]
            #     all_score = []
            #     for c in all_candid:
            #         ws = get_score(wc, c)
            #         all_score.append(ws)
            # else:
            all_candid = list(set(wc[0] + ac[0]))
            all_score = []
            for c in all_candid:
                ws = get_score(wc, c)
                ass = get_score(ac, c)
                s = alpha * ws + (1 - alpha) * ass
                all_score.append(s)
            sort_idx =  np.argsort(np.array(all_score))[::-1]
            sort_idx = sort_idx[:30]
            all_score = [all_score[x] for x in sort_idx]
            all_candid = [all_candid[x] for x in sort_idx]
            all_merge = " || ".join([f"{x} | {y}" for x, y in zip(all_candid, all_score)])

            fo1.write("{} ||| {}\n".format(m, " || ".join(all_candid)))
            fo2.write("{} ||| {}\n".format(m, all_merge))



if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) >= 2 else 0.5
    beta = float(sys.argv[2]) if len(sys.argv) >= 3 else 1
    langs = ["il5", "il6", "il9", "il10"]
    plangs = ["am", "id", "tl", "hi"]
    base_plangs = ["am", "id", "tl", "hi"]
    representation = ["ipa", "graph", "ipa", "ipa"]
    base_representations = ["ipa", "graph", "ipa", "ipa"]

    result_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/results"
    save_path = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/"
    query_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    models = ["base", "ours"]
    file_map = {"base":"me_pivot_en-{}_en-{}_ee_lstm-cosine-hinge_{}.id",
                "ours":"me_pivot_en-{}_en-{}_ee-me_char-cosine-hinge_aka_{}.id"}
    save_file_map = {"base": "base", "ours": "ours"}
    for lang, plang, reprep, base_plang, base_representation in zip(langs, plangs, representation, base_plangs, base_representations):
        # if lang != "il6":
        #     continue
        print("==========================")
        for model_idx, model in enumerate(models):
            # print(model)
            if model_idx == 0: #base
                cur_plang = base_plang
                cur_reprep = base_representation
            else:
                cur_plang = plang
                cur_reprep = reprep
            print(lang, cur_plang, cur_reprep)
            mention_file = os.path.join(query_path, f"me_test_en-{lang}_links")
            mention_list, answer_list = extract_mention(mention_file)
            generate_candid_file = os.path.join(result_path, file_map[model].format(lang, cur_plang, cur_reprep))
            save_file = os.path.join(save_path, f"{lang}_dataset", "map", "{}_{}_mention".format(lang, save_file_map[model]))
            load_result(generate_candid_file, save_file, mention_list, beta)
            calc_recall(mention_file, save_file + "_meprior")

            print("wiki")
            wiki_file = os.path.join(save_path, f"{lang}_dataset", "map", "{}_wiki_mention_meprior".format(lang))
            calc_recall(mention_file, wiki_file)

            print("merge")
            merge_save_file = os.path.join(save_path, f"{lang}_dataset", "map", "{}_{}_merge_mention".format(lang, save_file_map[model]))
            merge_candidates(wiki_file, save_file + "_meprior", mention_list, merge_save_file, alpha)
            calc_recall(mention_file, merge_save_file + "_meprior")

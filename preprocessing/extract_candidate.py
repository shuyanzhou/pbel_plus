import os
import numpy as np
lang = "il9"
mentions = []

mention_file = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/raw_data/lrl/mentions/{}_all_mentions_recover".format(lang)
with open(mention_file, "r", encoding="utf-8") as f:
    for line in f:
        mentions.append(line.strip())

candid_file = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/raw_data/lrl/mentions/{}.pivot.result.id".format(lang)
save_file = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/{}_dataset/map/{}_pivot_mention".format(lang, lang)

with open(candid_file, "r", encoding="utf-8") as fin:
    with open(save_file + "_meprior", "w+", encoding="utf-8") as fout:
        with open(save_file + ".map", "w+", encoding="utf-8") as fmap:
            for idx, line in enumerate(fin):
                tks = line.strip().split(" ||| ")
                tks[0] = mentions[idx]
                entity_info = tks[1].split(" || ")
                entity_info = [x.split(" | ") for x in entity_info]
                entity_id = np.array([int(x[0]) for x in entity_info])
                scores = np.array([float(x[1]) for x in entity_info])
                resort_idx = np.argsort(scores)[::-1]
                resort_idx = resort_idx[:30]
                scores = scores[resort_idx]
                entity_id = entity_id[resort_idx]
                scores = [str(x) for x in scores]
                entity_id = [str(x) for x in entity_id]
                entity_info = [" | ".join([str(x), str(y)]) for x, y in zip(entity_id, scores)]
                entity_info = " || ".join(entity_info)
                fout.write("{} ||| {}\n".format(tks[0], entity_info))
                fmap.write("{} ||| {}\n".format(tks[0], " || ".join(entity_id)))
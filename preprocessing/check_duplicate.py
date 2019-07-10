import os
from collections import defaultdict
base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
for lang in ["hi", "am", "th", "so", "rn"]:
    # with open(os.path.join(base_path, f"ee_train_en-{lang}_links"), "r", encoding="utf-8") as f:
    #     print(lang)
    #     all = defaultdict(int)
    #     for line in f:
    #         if line in all:
    #             print(line)
    #         all[line.strip()] += 1
    #     print(len(all))
    #
    #     error = 0
    #     for k, v in all.items():
    #         if v > 1:
    #             error += 1
    #     print(error)
    print(lang)
    with open(os.path.join(base_path, f"ee_train_en-{lang}_links"), "r", encoding="utf-8") as f, \
        open(os.path.join(base_path, "ndp_mention", f"mend_train_en-{lang}_links")) as fnp:
        all = {}
        for line in fnp:
            tks=line.strip().split(" ||| ")
            all[(tks[1] + "," + tks[2])] = 1
        print(len(all))
        for line in f:
            tks = line.strip().split(" ||| ")
            if tks[1] + "," + tks[2] not in all:
                all[(tks[1] + "," + tks[2])] = 1
        print(len(all))
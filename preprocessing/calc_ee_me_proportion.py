import os
import sys

# lang = sys.argv[1]
# data = sys.argv[2]
for lang in ["am", "so", "hi", "th", "ta", "rn"]:
    print("====================================")
    for data in ["ee_train", "ee-me_train", "me_val"]:
        # lang_link_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{}_links".format(lang)
        fname = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/{}_en-{}_links".format(data, lang)
        print(lang, data)
        # lang_link_map = {}
        # with open(lang_link_file, "r", encoding="utf-8") as f:
        #     for line in f:
        #         tks = line.strip().split(" ||| ")
        #         if len(tks) == 3:
        #             lang_link_map[tks[1]] = tks[2]
        # print(len(lang_link_map))


        # tot = 0
        # exact_match = 0
        # error = 0
        # longer = 0
        # shorter = 0
        # with open(fname, "r", encoding="utf-8") as f:
        #     for line in f:
        #         tks = line.strip().split(" ||| ")
        #         tot += 1
        #         if tks[1] not in lang_link_map:
        #             error += 1
        #             continue
        #         if tks[2] == lang_link_map[tks[1]]:
        #             exact_match += 1
        #         else:
        #             if len(tks[2]) >  len(lang_link_map[tks[1]]):
        #                 longer += 1
        #             else:
        #                 shorter += 1
        #         if data=="me_val" and tot >= 2000:
        #             break
        # print(error, tot, exact_match / float(tot), longer / float(tot), shorter / float(tot))

        tot = 0
        exact_match = 0
        error = 0
        longer = 0
        shorter = 0
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                tks = line.strip().split(" ||| ")
                tot += 1
                en = tks[1]
                hl = tks[2]
                if len(en.split()) == len(hl.split()):
                    exact_match += 1
        print(error, exact_match, tot, exact_match / float(tot), longer / float(tot), shorter / float(tot))


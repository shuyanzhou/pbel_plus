import os
import sys
from collections import defaultdict

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"

def get_ngram(string, ngram_list=(2, 3, 4, 5)):
    all_ngrams = []
    char_list = [START_SYMBOL] + list(string) + [END_SYMBOL]
    for n in ngram_list:
        cur_ngram = zip(*[char_list[i:] for i in range(n)])
        cur_ngram = ["".join(x) for x in cur_ngram]
        all_ngrams += cur_ngram
    return all_ngrams


def calc_n_gram_overlap(plang, tlang, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    pchar = defaultdict(int)
    tchar = []

    data_path =  os.path.join(base_path, f"ee-me_train_en-{plang}_links{encode}")
    tot = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            cur_ngram = get_ngram(s)
            for ngram in cur_ngram:
                pchar[ngram] += 1
            tot += 1
    print(tot)

    data_path = os.path.join(base_path, f"me_test_en-{tlang}_links{encode}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            tchar += get_ngram(s)

    # pchar = list(set(pchar))
    # pchar = {k: 1 for k in pchar}
    all_pchar = sum(pchar.values())
    # pchar = {k: v/all_pchar for k, v in pchar.items()}
    tot = len(tchar)
    unk = 0
    occur = 0
    for c in tchar:
        if c in pchar:
            occur += pchar[c]
        else:
            unk += 1
    print(occur)
    print((tot - unk) / tot)
    return (tot - unk) / tot


for plang in ["th", "am", "hi", "rn", "so", "ta"]:
    for tlang in ["lo", "mr", "te", "il5", "il6", "il9", "il10"]:
        print("====================================")
        print(plang, tlang)
        calc_n_gram_overlap(plang, tlang, ".ipa")

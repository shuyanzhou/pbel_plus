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


def extract_plang_ngram(plang, encode):
    pchar = defaultdict(float)

    save_train_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{plang}_processed"
    data_path = os.path.join(save_train_path, f"ee-me_train_en-{plang}_links{encode}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) < 3:
                continue
            s = tks[2]
            cur_ngram = get_ngram(s)
            for ngram in cur_ngram:
                pchar[ngram] += 1

    tot = sum(list(pchar.values()))
    # calculate the probability
    for k, v in pchar.items():
        pchar[k] = v / tot
    return pchar

def extract_tlang_ngram(tlang, encode):
    tchar = defaultdict(float)
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    data_path = os.path.join(base_path, f"me_test_en-{tlang}_links{encode}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            cur_ngram = get_ngram(s)
            for ngram in cur_ngram:
                tchar[ngram] += 1
    tot = sum(list(tchar.values()))
    # calculate the probability
    for k, v in tchar.items():
        tchar[k] = v / tot
    return tchar

def calc_n_gram_overlap(pchar, tchar):
    tot = len(tchar)
    unk = 0
    occur = 0
    for c in tchar:
        if c in pchar:
            # occur += min(pchar[c], tchar[c])
            occur += tchar[c]
        else:
            unk += 1
    # return (tot - unk) / tot
    return occur

all_plangs = ["ti", "ny", "om", "xh", "zu", "tpi", "rw", "mt", "lo", "ha", "ug", "sn", "so",
                  "tk", "ilo", "am", "ckb", "ku", "pa", "yo", "my", "sw", "mr", "jv", "te", "bn",
                  "tl", "ky", "tg", "ta", "uz", "hi", "th", "az", "kk", "ms", "tr", "ro", "hu",
                  "id", "ca", "ar", "uk", "pt", "vi", "pl", "es", "it", "ru", "nl",
                  "fr", "de", "sv", "ceb"]
all_tlangs = ["lo", "mr", "te", "il5", "il6", "il9", "il10"]

all_tlang_ngram = []
all_tlang_ipa_ngram = []
all_plang_ngram = []
all_plang_ipa_ngram = []
for plang in all_plangs:
    pchar = extract_plang_ngram(plang, "")
    pchar_ipa = extract_plang_ngram(plang, ".ipa")
    all_plang_ngram.append(pchar)
    all_plang_ipa_ngram.append(pchar_ipa)

for tlang in all_tlangs:
    tchar = extract_tlang_ngram(tlang, "")
    tchar_ipa = extract_tlang_ngram(tlang, ".ipa")
    all_tlang_ngram.append(tchar)
    all_tlang_ipa_ngram.append(tchar_ipa)

print("done")
for pid, plang in enumerate(all_plangs):
    for tid, tlang in enumerate(all_tlangs):
        cur_tchar = all_tlang_ngram[tid]
        cur_tchar_ipa = all_tlang_ipa_ngram[tid]
        cur_pchar = all_plang_ngram[pid]
        cur_pchar_ipa = all_plang_ipa_ngram[pid]

        overlap = calc_n_gram_overlap(cur_pchar, cur_tchar)
        info = [tlang, plang, str(overlap),"graph"]
        print("\t".join(info))

        overlap = calc_n_gram_overlap(cur_pchar_ipa, cur_tchar_ipa)
        info = [tlang, plang, str(overlap), "ipa"]
        print("\t".join(info))

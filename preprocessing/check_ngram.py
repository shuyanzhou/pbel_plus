'''
this code is used to extract n-gram and their norms from one language
and check their distribution on another language
'''

import os
import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_c2i_map(plang, model):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/c2i_maps"
    map_path =  os.path.join(base_path, f"en-{plang}_{model}_src.pkl")
    with open(map_path, "rb") as f:
        c2i_map = pickle.load(f)
        i2c_map = {v:k for k, v in c2i_map.items()}
    print("c2i map:", len(c2i_map))
    return c2i_map, i2c_map


def load_n_gram_embedding(plang, model):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/models"
    model_path = os.path.join(base_path, f"en-{plang}_{model}_best.tar")
    model_info = torch.load(model_path, map_location=DEVICE)
    model = model_info["model_state_dict"]
    src_lookup = model["src_lookup.weight"]
    lookup_norm = torch.norm(src_lookup, dim=1)
    # print("look up shape:", lookup_norm.shape)
    return lookup_norm

def get_ngram(string, ngram_list=(2, 3, 4, 5)):
    all_ngrams = []
    char_list = [START_SYMBOL] + list(string) + [END_SYMBOL]
    for n in ngram_list:
        cur_ngram = zip(*[char_list[i:] for i in range(n)])
        cur_ngram = ["".join(x) for x in cur_ngram]
        all_ngrams += cur_ngram
    return all_ngrams

def extract_n_gram(tlang, test_data, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    data_path = os.path.join(base_path, f"{test_data}_en-{tlang}_links{encode}")

    all_ngram = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            all_ngram += get_ngram(s)

    return all_ngram

def calc_distribution(c2i_map, n_gram_embedding, test_n_gram, tlang, max_norm=None):
    norm = []
    unk = 0
    tot = 0
    for n_gram in test_n_gram:
        if n_gram in c2i_map:
            norm.append(n_gram_embedding[c2i_map[n_gram]].cpu())
        else:
            norm.append(0.0)
            unk += 1
        tot += 1
    norm = np.array(norm)

    print("unk:",unk, "tot:", tot, "overlap:", (tot -  unk) / tot)

    if max_norm is None:
        max_norm  = np.max(norm)

    plt.hist(norm, bins=np.linspace(0, max_norm, 40), weights=np.ones(len(norm)) / len(norm), label=tlang, alpha=0.5)
    return np.max(norm)

def calc_character_overlap(plang, tlang, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    pchar = []
    tchar = []
    for data, lang, all_char in zip(["ee-me_train", "me_test"], [plang, tlang], [pchar, tchar]):
        data_path = os.path.join(base_path, f"{data}_en-{lang}_links{encode}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                tks = line.strip().split(" ||| ")
                s = tks[2]
                all_char += list(s)

    pchar = list(set(pchar))
    pchar = {k: 1 for k in pchar}
    tot = len(tchar)
    unk = 0
    for c in tchar:
        if c not in pchar:
            unk += 1
    print(unk, tot, (tot - unk) / tot)

def calc_n_gram_overlap(plang, tlang, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    pchar = []
    tchar = []
    for data, lang, all_char in zip(["ee-me_train", "me_test"], [plang, tlang], [pchar, tchar]):
        data_path = os.path.join(base_path, f"{data}_en-{lang}_links{encode}")
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                tks = line.strip().split(" ||| ")
                s = tks[2]
                all_char += get_ngram(s)
    pchar = list(set(pchar))
    pchar = {k: 1 for k in pchar}
    tot = len(tchar)
    unk = 0
    for c in tchar:
        if c not in pchar:
            unk += 1
    print(unk, tot, (tot - unk) / tot)

if __name__ == "__main__":
    all_plang = ["am", "hi", "hi", "hi", "th", "so", "rn", "so"]
    all_tlang = ["il5", "mr", "te", "il10", "lo", "il6", "il9", "il9"]
    all_other = ["hi", "hi", "ta", "am", "hi", "rn", "so", "so"]
    all_encode = ["ipa", "graph", "ipa", "ipa", "ipa", "graph", "graph", "graph"]
    test_data = ["ee-me_train", "me_test"]
    for idx, (plang, tlang, other, encode) in enumerate(zip(all_plang, all_tlang, all_other, all_encode)):
        print("===================")
        print(plang, tlang, other, encode)
        # for i in range(2):
        #     cur_tlang = [all_plang, all_tlang][i][idx]
        #     cur_test_data = test_data[i]
        #     model = f"ee-me_char-cosine-hinge_{encode}"
        #     c2i_map, _ = load_c2i_map(plang, model)
        #     n_gram_embedding = load_n_gram_embedding(plang, model)
        #     test_n_gram = extract_n_gram(cur_tlang, cur_test_data, encode)
        #     if i == 0:
        #         max_norm = None
        #     max_norm = calc_distribution(c2i_map, n_gram_embedding, test_n_gram, cur_tlang, max_norm)
        # plt.legend()
        # plt.savefig(f"/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/hist/{tlang}-{all_plang[idx]}.hist.png")
        # plt.clf()
        #
        # for i in range(2):
        #     cur_tlang = [all_other, all_tlang][i][idx]
        #     cur_test_data = test_data[i]
        #     model = f"ee-me_char-cosine-hinge_{encode}"
        #     c2i_map, _ = load_c2i_map(other, model)
        #     n_gram_embedding = load_n_gram_embedding(other, model)
        #     test_n_gram = extract_n_gram(cur_tlang, cur_test_data, encode)
        #     if i == 0:
        #         max_norm = None
        #     max_norm = calc_distribution(c2i_map, n_gram_embedding, test_n_gram, cur_tlang, max_norm)
        # plt.legend()
        # plt.savefig(f"/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/hist/{tlang}-{all_other[idx]}.hist.png")
        # plt.clf()
        calc_n_gram_overlap(plang, tlang, encode)
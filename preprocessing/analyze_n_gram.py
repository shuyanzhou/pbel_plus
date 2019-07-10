'''
this code is mainly use to extract n-grams with highest norm
'''
import torch
import numpy as np
import os
import sys
import pickle

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_c2i_map(plang, model, side):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/c2i_maps"
    map_path =  os.path.join(base_path, f"en-{plang}_{model}_{side}.pkl")
    with open(map_path, "rb") as f:
        c2i_map = pickle.load(f)
        i2c_map = {v:k for k, v in c2i_map.items()}
    print("c2i map:", len(c2i_map))
    return c2i_map, i2c_map


def load_n_gram_embedding(plang, model, side):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/models"
    model_path = os.path.join(base_path, f"en-{plang}_{model}_best.tar")
    model_info = torch.load(model_path, map_location=DEVICE)
    model = model_info["model_state_dict"]
    src_lookup = model[f"{side}_lookup.weight"]
    lookup_norm = torch.norm(src_lookup, dim=1)
    # print("look up shape:", lookup_norm.shape)
    return src_lookup, lookup_norm

def get_ngram(string, ngram_list=(2, 3, 4, 5)):
    all_ngrams = []
    char_list = [START_SYMBOL] + list(string) + [END_SYMBOL]
    for n in ngram_list:
        cur_ngram = zip(*[char_list[i:] for i in range(n)])
        cur_ngram = ["".join(x) for x in cur_ngram]
        all_ngrams += cur_ngram
    return all_ngrams

def retrieve_n_gram_lines(c2i_map, i2c_map, target_n_gram_idx, tlang, test_data, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    data_path = os.path.join(base_path, f"{test_data}_en-{tlang}_links{encode}")

    all_target_lines = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            cur_n_gram = get_ngram(s)
            n_gram_idx = [c2i_map.get(x, -1) for x in cur_n_gram]
            cur_target = list(set(n_gram_idx).intersection(set(target_n_gram_idx)))
            cur_target = [i2c_map[x] for x in cur_target]
            if len(cur_target) != 0:
                all_target_lines.append([tks[1], tks[2], cur_target])

    return all_target_lines


def print_knn(trg_idx, topk_idx, src_i2c_map, trg_i2c_map):
    for idx, trg in enumerate(trg_idx):
        print(f"{trg_i2c_map[trg.item()]}")
        hrl = [src_i2c_map[x.item()] for x in topk_idx[idx]]
        hrl = "\t".join(hrl)
        print(f"{hrl}")

def find_knn(trg_lookup, src_lookup, bigram_idx, t=20, k=5):
    trg_size = trg_lookup.shape[0]
    # random_idx = torch.randint(0, trg_size, (t, ))
    random_idx = np.random.choice(bigram_idx, (t, ))
    random_idx = torch.from_numpy(random_idx)
    trg_trg = trg_lookup[random_idx]

    src_norm = torch.norm(src_lookup, dim=1, keepdim=True)
    src_norm_encoded = src_lookup / src_norm
    trg_norm = torch.norm(trg_trg, dim=1, keepdim=True)
    trg_norm_encoded = trg_trg / trg_norm
    similarity = torch.matmul(trg_norm_encoded, torch.transpose(src_norm_encoded, 1, 0))
    _, topk_idx = torch.topk(similarity, k=k, dim=1)
    return random_idx, topk_idx


def main_check_highest_norm():
    k = 20
    all_plang = ["am", "hi", "hi", "hi", "th", "so", "rn"]
    all_tlang = ["il5", "mr", "te", "il10", "lo", "il6", "il9"]
    all_other = ["hi", "hi", "ta", "am", "hi", "rn", "so"]
    all_encode = ["ipa", "graph", "ipa", "ipa", "ipa", "graph", "graph"]
    test_data = ["ee-me_train", "me_test"]
    for idx, (plang, tlang, other, encode) in enumerate(zip(all_plang, all_tlang, all_other, all_encode)):
        print(" =====================================================")
        print(plang, tlang, encode)
        model = f"ee-me_char-cosine-hinge_{encode}"
        c2i_map, i2c_map = load_c2i_map(plang, model, "src")
        _, norm_lookup = load_n_gram_embedding(plang, model, "src")
        _, topk_idx = torch.topk(norm_lookup, k=k, dim=0)
        topk_idx = topk_idx.cpu().numpy().tolist()
        print("============top k n-grams=============")
        print(topk_idx)
        print([i2c_map[x] for x in topk_idx])
        all_target_lines = retrieve_n_gram_lines(c2i_map, i2c_map, topk_idx, plang, "ee-me_train", encode)
        for l in all_target_lines:
            print(l)

def main_find_knn():
    all_plang = ["am", "hi", "hi", "hi", "th", "so", "rn"]
    all_tlang = ["il5", "mr", "te", "il10", "lo", "il6", "il9"]
    all_other = ["hi", "hi", "ta", "am", "hi", "rn", "so"]
    all_encode = ["ipa", "graph", "ipa", "ipa", "ipa", "graph", "graph"]
    test_data = ["ee-me_train", "me_test"]
    for idx, (plang, tlang, other, encode) in enumerate(zip(all_plang, all_tlang, all_other, all_encode)):
        print(" =====================================================")
        print(plang, tlang, encode)
        model = f"ee-me_char-cosine-hinge_{encode}"
        _, src_i2c_map = load_c2i_map(plang, model, "src")
        _, trg_i2c_map = load_c2i_map(plang, model, "trg")

        src_lookup, _ = load_n_gram_embedding(plang, model, "src")
        trg_lookup, trg_norm = load_n_gram_embedding(plang, model, "trg")

        _, bigram_idx = torch.topk(trg_norm, k=300, dim=0)
        bigram_idx = bigram_idx.cpu().numpy().tolist()

        trg_idx, src_topk = find_knn(trg_lookup, src_lookup, bigram_idx)
        print_knn(trg_idx, src_topk, src_i2c_map, trg_i2c_map)

if __name__ == "__main__":
    main_find_knn()
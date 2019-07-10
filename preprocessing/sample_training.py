import os
import sys
import random
from collections import Counter, defaultdict

START_SYMBOL = "<s>"
END_SYMBOL = "</s>"

# random.seed(1234)
def sample_data(lang, tlang, encode, selected_n_gram, prefix="ee-me", k=10000):
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"

    all_data = []
    random_data = []
    with open(os.path.join(base_path, f"{prefix}_train_en-{lang}_links{encode}"), encoding="utf-8") as f:
        for line in f:
            all_data.append(line.strip())
    random.shuffle(all_data)

    for line in all_data:
        tks = line.strip().split(" ||| ")
        cur_n_gram = get_ngram(tks[2])
        good = all([x not in selected_n_gram for x in cur_n_gram])
        if good:
            random_data.append(line.strip())
        if len(random_data) >= k:
            break

    print(f"number of data: {len(random_data)}")
    overlap = calc_n_gram_overlap(random_data, tlang, encode)

    with open(os.path.join(base_path, "sample_v2", f"{prefix}_train_en-{lang}_links_{str(overlap)[:4]}{encode}"), "w+", encoding="utf-8") as f:
        for d in random_data:
            f.write(d + "\n")


def get_ngram(string, ngram_list=(2, 3, 4, 5)):
    all_ngrams = []
    char_list = [START_SYMBOL] + list(string) + [END_SYMBOL]
    for n in ngram_list:
        cur_ngram = zip(*[char_list[i:] for i in range(n)])
        cur_ngram = ["".join(x) for x in cur_ngram]
        all_ngrams += cur_ngram
    return all_ngrams

def calc_n_gram_overlap(plang_data, tlang, encode):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    pchar = []
    tchar = []

    for line in plang_data:
        tks = line.strip().split(" ||| ")
        s = tks[2]
        pchar += get_ngram(s)

    data_path = os.path.join(base_path, f"me_test_en-{tlang}_links{encode}")
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            s = tks[2]
            tchar += get_ngram(s)

    pchar = list(set(pchar))
    pchar = {k: 1 for k in pchar}
    tot = len(tchar)
    unk = 0
    for c in tchar:
        if c not in pchar:
            unk += 1
    print(unk, tot, (tot - unk) / tot)
    return (tot - unk) / tot


def get_n_gram_proportion(lang, encode, p):
    n_gram_counter = Counter()
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    with open(os.path.join(base_path, f"me_test_en-{lang}_links{encode}"), "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            cur_n_gram = get_ngram(tks[2])
            for n in cur_n_gram:
                n_gram_counter[n] += 1

    tot = sum(list(n_gram_counter.values()))
    print(f"number of n-grams: {tot}")
    n_gram_p = {k: v/tot for k, v in n_gram_counter.items()}
    all_n_gram = [k for k in n_gram_p.keys()]
    n_gram_p = [n_gram_p[k] for k in all_n_gram]
    n_gram_count = [n_gram_counter[k] for k in all_n_gram]
    accumulate = [sum(n_gram_p[:i]) for i in range(len(n_gram_p))]
    assert abs(accumulate[-1] - 1) <= 0.001, accumulate[-1]

    for i in range(len(accumulate) - 1):
        if accumulate[i] <= p and accumulate[i + 1] >= p:
            selected_n_gram = all_n_gram[i:]
            selected_count = n_gram_count[i:]
            selected_p = n_gram_p[i:]
            selected_n_gram = {k: selected_count[i] for i, k in enumerate(selected_n_gram)}
            assert abs(sum(selected_p) - (1 - accumulate[i])) <= 0.0001
            print(f"proportion: {1 - accumulate[i]}, with {sum(selected_n_gram.values())}/{len(selected_n_gram)} n-grams inside")

    return selected_n_gram

    # all_n_grams = list(n_gram_counter.keys())
    # num = int(len(all_n_grams) * p)
    # random.shuffle(all_n_grams)
    # selected_n_gram = {k: n_gram_counter[k] for k in all_n_grams[:num]}
    # print(f"{p} n-grams count for {sum(selected_n_gram.values())/sum(n_gram_counter.values())}")
    # return selected_n_gram


def sample_training(all_plang, all_tlang, encode, n = 10000):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"

    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""

    for plang in all_plang:
        print(plang)
        all_data = []
        with open(os.path.join(base_path, f"ee-me_train_en-{plang}_links{encode}"), encoding="utf-8") as f:
            for line in f:
                all_data.append(line.strip())
            random.shuffle(all_data)
        cur_n = min(n, len(all_data))
        print(cur_n)
        random_data = all_data[:cur_n]
        for tlang in all_tlang:
            print(tlang)
            overlap = calc_n_gram_overlap(random_data, tlang, encode)

        with open(os.path.join(base_path, "sample_data", f"ee-me_train_en-{plang}_links_{n}{encode}"), "w+",encoding="utf-8") as f:
            for d in random_data:
                f.write(d + "\n")
        return overlap

if __name__ == "__main__":
    # plang = sys.argv[1]
    # tlang = sys.argv[2]
    # encode = sys.argv[3]
    # p = float(sys.argv[4])
    # num = int(sys.argv[5])
    # selected_n_gram = get_n_gram_proportion(tlang, encode, p)
    # sample_data(plang, tlang, encode, selected_n_gram, k=num)
    all_plang = ["hi"]
    all_tlang = ["mr"]
    encode = ""
    overlap = 0
    while overlap < 0.746:
        overlap = sample_training(all_plang, all_tlang, encode, n=10000)
import os
import sys

lang = sys.argv[1]
data = sys.argv[2]

base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"

with open(os.path.join(base_path, f"{data}_en-{lang}_links"), "r", encoding="utf-8") as f1, open(os.path.join(base_path, f"{data}_en-{lang}_links.ipa"), "r", encoding="utf-8") as f2:
    with open(os.path.join(base_path, f"{data}_en-{lang}_links.merge"), "w+", encoding="utf-8") as fout:
        for l1, l2 in zip(f1, f2):
            tks1 = l1.split(" ||| ")
            tks2 = l2.split(" ||| ")
            assert tks1[0] == tks2[0]
            en=tks1[1]
            en_ipa = tks2[1]
            other = tks1[2]
            other_ipa = tks2[2]
            fout.write(" ||| ".join([tks1[0], en, en_ipa, other, other_ipa]) + "\n")

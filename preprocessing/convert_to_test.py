import os
import epitran
import sys

lang = "il10"
ipa = True
base_path = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/raw_data/lrl/mentions/{}_all_mentions_recover".format(lang)

epi = epitran.Epitran("sin-Sinh")
with open(base_path, "r", encoding="utf-8") as f:
    with open(base_path + "_processed", "w+", encoding="utf-8") as fout:
        for line in f:
            mention = line.strip()
            if ipa:
                mention = epi.transliterate(mention)
            fout.write("-1 ||| -1 ||| " + mention + "\n")

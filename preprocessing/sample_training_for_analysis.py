import os
import sys
import epitran
import random

epitran_map = {"hi": "hin-Deva",
               "am": "amh-Ethi",
               "th": "tha-Thai",
               "tr": "tur-Latn",
               "ta": "tam-Taml",
               "id": "ind-Latn",
               "mr": "mar-Deva",
               "en": "eng-Latn",
               "ti": "tir-Ethi",
               "te": "tel-Telu",
               "lo": "lao-Laoo",
               "om": "orm-Latn",
               "kw": "kin-Latn",
               "si": "sin-Sinh",
               "il10": "sin-Sinh",
               "il10-all": "sin-Sinh",
               "il5": "tir-Ethi",
               "il5-all": "tir-Ethi",
               "il6": "som-Latn",
               "il6-all": "som-Latn",
               "il9": "run-Latn",
               "il9-all": "run-Latn"}

def sample_data(lang, encode):
    if encode == "ipa":
        encode = ".ipa"
    else:
        encode = ""
    if lang in ["hi", "ta", "th", "am"]:
        epi = epitran.Epitran(epitran_map[lang])
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    file_name = os.path.join(base_path, f"ee_train_en-{lang}_links{encode}")
    save_name = os.path.join(base_path, "train_analysis", f"ee_train_en-{lang}_links.tsv")
    with open(file_name, "r", encoding="utf-8") as f, open(save_name, "w+", encoding="utf-8") as fout:
        all_data = []
        for line in f:
            all_data.append(line.strip())
        random.shuffle(all_data)

        for d in all_data[:100]:
            tks = d.split(" ||| ")
            src_mention = tks[2]
            if lang in ["hi", "ta", "th", "am"]:
                ipa = epi.transliterate(src_mention)
            else:
                ipa = src_mention
            tks = tks[:3] + [ipa]
            fout.write("\t".join(tks) + "\n")


if __name__ == "__main__":
    for lang in ["hi", "ta", "th", "am", "so", "rn"]:
        sample_data(lang, "graph")
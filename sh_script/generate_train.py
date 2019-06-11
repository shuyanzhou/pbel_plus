import sys
import os
os.chdir("/home/shuyanzh/workshop/pbel/sh_script")

encode = ["graph", "ipa"]
file_suffix = ["", ".ipa"]
file_prefix = ["", "ipa_"]
file_name = ["graph", "ipa"]

all_lang = [["hi","am", "id", "so", "rn"],
            ["th", "tr", "ta", "am", "hi"]]

with open("./train_template.sh", "r", encoding="utf-8") as f:
    t = f.read()
    for i in range(2):
        e = encode[i]
        fs = file_suffix[i]
        fp = file_prefix[i]
        fn = file_name[i]

        with open(f"train_template_{fn}.sh", "w+", encoding="utf-8") as fout:
            cur_t = t.replace("ENCODE", e).replace("FILE_SUFFIX", fs).replace("FILE_PREFIX", fp)
            fout.write(cur_t)

for i in range(2):
    fn = file_name[i]
    with open(f"train_template_{fn}.sh", "r", encoding="utf-8") as f:
        t = f.read()
    for lang in all_lang[i]:
        with open(f"train_{lang}_{fn}.sh", "w+", encoding="utf-8") as fout:
            cur_t = t.replace("lang", lang)
            fout.write(cur_t)


import sys
import os
os.chdir("/home/shuyanzh/workshop/pbel/sh_script")

encode = ["graph", "ipa"]
file_suffix = ["", ".ipa"]
file_prefix = ["", "ipa_"]
file_name = ["graph", "ipa"]

all_lang = [
            [
                ["hi", "am", "id", "so", "rn"],
                ["mr", "ti", "jv", "il6", "il9"]
            ],
            [
                ["th", "am", "am", "hi", "hi"],
                ["lo", "ti", "il5", "il10", "te"]
            ]
        ]

with open("./test_template.sh", "r", encoding="utf-8") as f:
    t = f.read()
    for i in range(2):
        e = encode[i]
        fs = file_suffix[i]
        fp = file_prefix[i]
        fn = file_name[i]

        with open(f"test_template_{fn}.sh", "w+", encoding="utf-8") as fout:
            cur_t = t.replace("ENCODE", e).replace("FILE_SUFFIX", fs).replace("FILE_PREFIX", fp)
            fout.write(cur_t)

for i in range(2):
    fn = file_name[i]
    with open(f"test_template_{fn}.sh", "r", encoding="utf-8") as f:
        t = f.read()

    lang = all_lang[i]
    for pl, tl in zip(*lang):
        with open(f"test_{tl}_{fn}.sh", "w+", encoding="utf-8") as fout:
            cur_t = t.replace("tlang", tl)
            cur_t = cur_t.replace("lang", pl)
            if "il" in tl:
                cur_t = cur_t.replace('declare -a all_test_data=("ee" "me")', 'declare -a all_test_data=("me")')
            fout.write(cur_t)


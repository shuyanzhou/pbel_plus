import sys
import os
os.chdir("/home/shuyanzh/workshop/pbel/sh_script")

encode = ["graph", "ipa"]
file_suffix = ["", ".ipa"]
file_prefix = ["", "ipa_"]
file_name = ["graph", "ipa"]

hl = ["th", "ta", "hi", "so", "rn", "am"]
ll = ["lo", "mr", "te", "il5", "il6", "il9", "il10"]

all_hl = []
for x in hl:
    for i in range(7):
        all_hl.append(x)

all_ll = []
for i in range(6):
    for x in ll:
        all_ll.append(x)

all_lang = [
            [
                ["hi", "am", "id", "so", "rn", "so", "rn"],
                ["mr", "il5", "jv", "il6", "il9", "il9", "il6"]
            ],
            # [
            #     ["th", "am", "am", "hi", "hi"],
            #     ["lo", "ti", "il5-all", "il10-all", "te"]
            # ]
            [
                all_hl,
                all_ll
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
        with open(f"test_{tl}-{pl}_{fn}.sh", "w+", encoding="utf-8") as fout:
            cur_t = t.replace("tlang", tl)
            cur_t = cur_t.replace("lang", pl)
            if "il" in tl:
                cur_t = cur_t.replace('declare -a all_test_data=("me" "ee")', 'declare -a all_test_data=("me")')
                cur_t = cur_t.replace('declare -a all_test_data=("ee" "me")', 'declare -a all_test_data=("me")')
            if "ti" in tl:
                cur_t = cur_t.replace('declare -a all_test_data=("me" "ee")', 'declare -a all_test_data=("ee")')
                cur_t = cur_t.replace('declare -a all_test_data=("ee" "me")', 'declare -a all_test_data=("ee")')
            fout.write(cur_t)


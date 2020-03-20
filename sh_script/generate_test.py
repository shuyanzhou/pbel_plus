import sys
import os
os.chdir("/home/shuyanzh/workshop/pbel/sh_script")

encode = ["graph", "ipa"]
file_suffix = ["", ".ipa"]
file_prefix = ["", "ipa_"]
file_name = ["graph", "ipa"]

hl = ["th", "ta", "hi", "so", "rn", "am", "id", "tl", "ceb", "sv", "de"]
ll = ["lo", "mr", "te", "il5", "il6", "il9", "il10", "st5", "st6", "st9", "st10"]

all_hl = []
for x in hl:
    for i in range(len(ll)):
        all_hl.append(x)

all_ll = []
for i in range(len(hl)):
    for x in ll:
        all_ll.append(x)

gp_hl = ["hi", "am", "id", "so", "rn", "so", "rn", "id", "sv", "id", "hi", "id"]
gp_ll = ["mr", "il5", "jv", "il6", "il9", "il9", "il6", "il6", "il6", "il6", "st10", "st6"]

st_hl = ["tiam", "tiam", "so", "om", "id", "rw", "rn", "ki", "sw", "tl", "si", "mr", "bn", "hi"]
#        5             6                 9                            10
st_ll = ["st5" for i in range(2)] + ["st6" for i in range(3)] + \
    ["st9" for i in range(5)] + ["st10" for i in range(4)]

ll_encode = ["graph", "ipa"] + ["graph" for i in range(3)] + \
    ["graph" for i in range(4)] + ["ipa"] + \
    ["graph"] + ["ipa" for i in range(3)]

assert len(st_hl) == len(st_ll) and len(st_hl) == len(ll_encode)

for ll, hi, ed in zip(st_ll, st_hl, ll_encode):
    print(f"sbatch sh_script/test_{ll}-{hi}_{ed}.sh")


all_lang = [
            # grapheme
            [
                # gp_hl,
                # gp_ll
                st_hl,
                st_ll
            ],
            # phoneme
            [
                # all_hl,
                # all_ll
                st_hl,
                st_ll
            ]
        ]

with open("./test_template_split_kb.sh", "r", encoding="utf-8") as f:
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


import sys
import os

lang = "th"
plang = "lo"
base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
test_file = os.path.join(base_path, "ee_test_en-{}_links".format(plang))
pivot_file = os.path.join(base_path, "en-{}_links".format(lang))
kb_file = os.path.join(base_path, "../kb_split/en_kb")

nop_acc = 0
p_acc = 0
tot = 0

p_map = {}
with open(pivot_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        p_map[tks[2]] = int(tks[0])

kb_map = {}
with open(kb_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        kb_map[tks[1]] = int(tks[0])

exist = 0
with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        tot += 1
        tks = line.strip().split(" ||| ")
        m = tks[2]
        e = int(tks[0])
        if m in kb_map or m in p_map:
            exist += 1
        if kb_map.get(m, "") == e:
            nop_acc += 1
        if p_map.get(m, "") == e:
            p_acc += 1

print(exist, nop_acc/tot, p_acc/tot, tot)
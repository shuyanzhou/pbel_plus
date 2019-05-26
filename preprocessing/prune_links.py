import os
import sys
from collections import Counter

lang = sys.argv[1]
link_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{}_links".format(lang)
save_file = link_file + "_prune"
kb_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
all_entity = Counter()
with open(kb_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        all_entity[tks[0]] += 1
print(len(all_entity))
with open(link_file, "r", encoding="utf-8") as f:
    with open(save_file, "w+", encoding="utf-8") as fout:
        for line in f:
            tks = line.strip().split(" ||| ")
            if tks[0] in all_entity:
                fout.write(line)
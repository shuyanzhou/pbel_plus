import os
import sys
import epitran

kb_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
with open(kb_file, "r", encoding="utf-8") as f:
    id_type_map = {}
    for line in f:
        tks = line.strip().split(" ||| ")
        id_type_map[tks[0]] = tks[2]

lang = sys.argv[1]
base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
for prefix in ["ee", "ee-me", "me"]:
    for part in ["train", "val", "test"]:
        cur_fname = os.path.join(base_path, f"{prefix}_{part}_en-{lang}_links")
        if os.path.exists(cur_fname):
            with open(cur_fname, "r", encoding="utf-8") as f:
                with open(cur_fname + ".type", "w+", encoding="utf-8") as fout:
                    for line in f:
                        tks = line.strip().split(" ||| ")
                        if len(tks) == 4:
                            fout.write(line)
                        else:
                            type = id_type_map[tks[0]]
                            fout.write(" ||| ".join(tks + [type]) + "\n")
            os.rename(cur_fname + ".type", cur_fname)
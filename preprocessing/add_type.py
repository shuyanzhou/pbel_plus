import os
import sys

kb_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
with open(kb_file, "r", encoding="utf-8") as f:
    id_type_map = {}
    for line in f:
        tks = line.strip().split(" ||| ")
        id_type_map[tks[0]] = tks[2]

all_langs = sys.argv[1].split(",")
base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
for lang in all_langs:
    for suffix in ["", ".ipa", ".mid.ipa", ".mid"]:
        with open(os.path.join(base_path, "unique_mend_ee_val_en-{}_links".format(lang) + suffix), "r", encoding="utf-8") as f:
            with open(os.path.join(base_path, "unique_mend_ee_val_en-{}_links".format(lang) + suffix) + ".type", "w+", encoding="utf-8") as fout:
                for line in f:
                    tks = line.strip().split(" ||| ")
                    if len(tks) == 4:
                        fout.write(line)
                    else:
                        type = id_type_map[tks[0]]
                        fout.write(" ||| ".join(tks + [type]) + "\n")
        os.rename(os.path.join(base_path, "unique_mend_ee_val_en-{}_links".format(lang) + suffix) + ".type",
                  os.path.join(base_path, "unique_mend_ee_val_en-{}_links".format(lang) + suffix))
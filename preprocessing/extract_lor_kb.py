import os


def extract_lor_kb():
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/lor_kb"
    file_name = os.path.join(base_path, "entities.tab")
    save_file = os.path.join(base_path, "lor_kb")
    with open(file_name, "r", encoding="utf-8") as f, open(save_file, "w+", encoding="utf-8") as fout:
        f.readline()
        for line in f:
            tks = line.strip().split("\t")
            info = [tks[2], tks[3], tks[1]]
            fout.write(" ||| ".join(info) + "\n")


def extract_kb_id_map():
    with open("/projects/tir2/users/shuyanzh/lorelei_data/pbel/lor_kb/lor_kb", "r", encoding="utf-8") as f:
        d = {}
        for line in f:
            tks = line.strip().split(" ||| ")
            d[tks[0]] = tks[1]
    return d
def extract_test_data(id_name_map):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/raw_data/lrl"
    base_save = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
    for lang in ["il5", "il6", "il9", "il10"]:
        print(lang)
        file_name = os.path.join(base_path, lang, f"{lang}_edl.tab_recover")
        save_path = os.path.join(base_save, f"me_test_en-{lang}-all_links")
        tot = 0
        with open(file_name, "r", encoding="utf-8") as f, open(save_path, "w+", encoding="utf-8") as fout:
            for line in f:
                tks = line.strip().split("\t")
                mention, kb_id, entity_type = tks[2], tks[4], tks[5]
                if kb_id not in id_name_map:
                    continue
                tot += 1
                entity_name = id_name_map[kb_id]
                # entity_name=[]
                # valid_kb_id=[]
                # for kid in kb_id.split("|"):
                #     if kid in id_name_map:
                #         entity_name.append(id_name_map[kid])
                #         valid_kb_id.append(kid)
                # if len(valid_kb_id) != 0:
                #     tot += 1
                fout.write(" ||| ".join([kb_id, entity_name, mention, entity_type]) + "\n")
                # else:
                #     print(kb_id)
        print(tot)

if __name__ == "__main__":
    # extract_lor_kb()
    id_name_map = extract_kb_id_map()
    extract_test_data(id_name_map)
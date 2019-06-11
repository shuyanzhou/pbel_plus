'''
this script is used to transfer data used in BURN model to test data used in PBEL model
'''

import os


def get_type_map(kb_file):
    with open(kb_file, "r", encoding="utf-8") as f:
        id_type_map = {}
        for line in f:
            tks = line.strip().split(" ||| ")
            id_type_map[tks[0]] = tks[2]
    return id_type_map

def generate_test_data(lang, id_type_map):
    base_path = "/projects/tir2/users/shuyanzh/lorelei_data/TAC-KBP/"
    ner_dir = os.path.join(base_path,  f"{lang}_dataset", "eval_docs_ner_official")
    eval_file = os.path.join(base_path,  f"{lang}_dataset", "evaluation_query")
    pbel_base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/"
    save_file = os.path.join(pbel_base_path, f"lor_test_en-{lang}_links")

    with open(eval_file, "r", encoding="utf-8") as fin, open(save_file, "w+", encoding="utf-8") as fout:
        tot = 0
        nan_num = 0
        for line in fin:
            tks = line.strip().split(" ||| ")
            docid, _, _, wiki_title, wiki_id, _ = tks
            wiki_title = wiki_title.split(" || ")
            wiki_id = wiki_id.split(" || ")
            with open(os.path.join(ner_dir, docid + ".txt"), "r", encoding="utf-8") as fdoc:
                mentions = fdoc.read()
                mentions = mentions.strip().split(" ||| ")
                assert len(mentions) == len(wiki_id) and len(mentions) == len(wiki_title)

            non_nil_idx = [idx for idx, x in enumerate(wiki_id) if x != "NIL"]
            wiki_title = [wiki_title[x] for x in non_nil_idx]
            wiki_id = [wiki_id[x] for x in non_nil_idx]
            mentions = [mentions[x] for x in non_nil_idx]
            wiki_type = [id_type_map.get(x, "NAN") for x in wiki_id]
            nan_num += sum([1 for x in wiki_type if x == "NAN"])
            for id, title, mention, type in zip(wiki_id, wiki_title, mentions, wiki_type):
                tot +=  1
                fout.write(f"{id} ||| {title} ||| {mention} ||| {type}\n")
    print(tot, nan_num)


if __name__ == "__main__":
    kb_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
    id_type_map = get_type_map(kb_file)
    for lang in ["il5", "il6", "il9", "il10"]:
        print(lang)
        generate_test_data(lang, id_type_map)
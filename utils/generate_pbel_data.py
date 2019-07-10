import os
import random
import argparse
from collections import defaultdict
# generate test file (no duplicate), generate new test_file that don't have overlap with training data
def unique_test(train_file: list, test_file, new_test_file):
    ftrain = []
    for fname in train_file:
        ftrain.append(open(fname, "r", encoding="utf-8"))
    ftest = open(test_file, "r", encoding="utf-8")
    fnew_test = open(new_test_file, "w+", encoding="utf-8")

    train_set = set()
    for f in ftrain:
        for line in f:
            line = line.strip().split(" ||| ")[:3]
            line = " ||| ".join(line)
            train_set.add(line.strip())
    for line in ftest:
        line = line.strip()
        if line not in train_set:
            fnew_test.write(line + "\n")

    for x in ftrain:
        x.close()
    ftest.close()
    fnew_test.close()


def generate_pbel_data(lang, me_file, saved_me_file, link_file, en_title_id_file, saved_file_path, split_file=True):
    '''
    :param me_file: mention in lang ||| entities in lang ||| frequency
    :param saved_me_file: mention in source language ||| a list of entity in English
           saved_me_file_split: en wikipedia id ||| entity in en wikipedia || mention in source language
    :param link_file: en wikipedia ID ||| en str ||| lang str
    :param en_title_id_file: en wikipedia ID ||| en str ||| misc. This is the filtered KB, only with name entities!
    :param split_file: split files to train dev test if True, else only one
    :param saved_file_path: the path to save generate files
    :return:
    '''
    lang_en_map = {}
    with open(link_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 3:
                continue
            _, en_str, other_str = tks
            lang_en_map[other_str] = en_str
    print("[INFO] link map: {:d}".format(len(lang_en_map)))

    en_title_id_map = {}
    with open(en_title_id_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 3:
                continue
            en_title_id_map[tks[1]] = tks[0]
    print("[INFO] en title id map: {:d}".format(len(en_title_id_map)))

    # mention and entity are in lang
    me_map = {}
    with open(me_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            mention, entities, frequency = tks
            entities = entities.split(" || ")
            frequency = frequency.split(" || ")
            me_map[mention] = entities
            # duplicate_entity = []
            # for e, f in zip(entities, frequency):
            #     duplicate_entity += [e for _ in range(int(f))]
            # me_map[mention] = duplicate_entity
    print("[INFO] mention entity map: {:d}".format(len(me_map)))

    all_pairs = []
    # mention in source language ||| a list of entity in English
    # en wikipedia id ||| entity in en wikipedia || mention in source language
    with open(saved_me_file, "w+", encoding="utf-8") as f, open(saved_me_file + ".split", "w+", encoding="utf-8") as fspl:
        for m, el in me_map.items():
            en_el = [lang_en_map.get(x, "<unk>") for x in el]
            en_el = [x for x in en_el if x != "<unk>"]
            en_id_el = [en_title_id_map.get(x, "<unk>") for x in en_el]
            assert len(en_el) == len(en_id_el)

            # filter entities that do not have entity-id map in PRUNED KB file
            pruned_en_el = [x for i, x in enumerate(en_el) if en_id_el[i] != "<unk>"]
            if len(pruned_en_el) != 0:
                f.write("{} ||| {}\n".format(m, " || ".join(list(set(pruned_en_el)))))

            for e, eid in zip(en_el, en_id_el):
                # star symbol somewhere
                if eid != "<unk>" and m != "*":
                    fspl.write("{} ||| {} ||| {}\n".format(eid, e, m))
                    all_pairs.append("{} ||| {} ||| {}\n".format(eid, e, m))
        print(f"[INFO] there are {len(all_pairs)} pairs")

    # generate train dev test files
    file_name = f"en-{lang}_links"
    random.shuffle(all_pairs)
    sample_num = len(all_pairs)
    if split_file:
        train = all_pairs[: int(sample_num * 0.7)]
        dev = all_pairs[int(sample_num * 0.7): int(sample_num * 0.9)]
        test = all_pairs[int(sample_num * 0.9):]
        for prefix, data in zip(["me_train", "me_dup_val", "me_dup_test"], [train, dev, test]):
            with open(os.path.join(saved_file_path, prefix + "_" + file_name), "w+", encoding="utf-8") as f:
                for d in data:
                    f.write(d)

        # remove data that already appears in the training data
        unique_test(train_file=[os.path.join(saved_file_path, "me_train_" + file_name),
                     os.path.join(saved_file_path, "ee_train_" + file_name)],
                    test_file=os.path.join(saved_file_path, "me_dup_val_" + file_name),
                    new_test_file=os.path.join(saved_file_path, "me_val_" + file_name))

        unique_test([os.path.join(saved_file_path, "me_train_" + file_name),
                     os.path.join(saved_file_path, "ee_train_" + file_name)],
                    os.path.join(saved_file_path, "me_dup_test_" + file_name),
                    os.path.join(saved_file_path, "me_test_" + file_name))

def save_mention_entity_string_frequency(me_file, new_me_file, title_id_file):
    id_title_map = {}
    with open(title_id_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            title, id = tks
            id_title_map[id] = title
    print("[INFO] title id map: {:d}".format(len(id_title_map)))

    me_map = defaultdict(list)
    with open(me_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            mention, entity_frequency = tks
            entity_frequency = entity_frequency.split(" || ")
            entity_frequency = [x.split(" | ") for x in entity_frequency]
            entity = [x[0] for x in entity_frequency]
            frequency = [x[1] for x in entity_frequency]
            me_map[mention] = [entity, frequency]
    print("[INFO] mention entity map: {:d}".format(len(me_map)))

    with open(new_me_file, "w+", encoding="utf-8") as fout:
        for m, (el, fl) in me_map.items():
            # delete all "<unk>"
            known_el = []
            known_fl = []
            for e, f in zip(el, fl):
                e_str = id_title_map.get(e, "<unk>")
                if e_str != "<unk>":
                    known_el.append(e_str)
                    known_fl.append(f)
            assert len(known_el) == len(known_fl)
            fout.write("{} ||| {} ||| {}\n".format(m, " || ".join(known_el), " || ".join(known_fl)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    args, _ = parser.parse_known_args()
    lang = args.lang

    base_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{lang}_processed/"
    me_file = os.path.join(base_path, "meprior")
    me_file_with_str = os.path.join(base_path, "meprior.str")
    me_file_with_en = os.path.join(base_path, "meprior.en.str")
    title_id_file = os.path.join(base_path, "title_id_map")

    en_link_file = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{lang}_links"
    en_title_id_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
    save_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{lang}_processed/"

    save_mention_entity_string_frequency(me_file, me_file_with_str, title_id_file)
    generate_pbel_data(lang, me_file_with_str, me_file_with_en, en_link_file, en_title_id_file, save_path)
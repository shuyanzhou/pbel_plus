'''
1.process anchor text to get a {anchor text: KB entry} dictionary
2.cal entity probability, cal entity probability given mention
3.entity id ||| entity title ||| a list of id that are in the same page ||| entity description

The input format of the data is a json file https://github.com/shuyanzhou/Annotated-WikiExtractor

!!! if the linked page don't have corresponding wikiepdia id, just DISCARD it !!!
'''

import json
import codecs
import os
import sys
import bz2
import io
import re
import math
import numpy as np
from decimal import Decimal
from collections import defaultdict
from urllib.parse import unquote
import functools
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer
import html
import random

print = functools.partial(print, flush=True)

class JsonProcessor():
    def __init__(self, raw_path, anchor_map_file, entity_page_file, eprior_file, meprior_file, log_file, load_redirect_map=True, redirect_file=None, wiki_dump = None, load_title_id=False, title_id_file=None):
        self.raw_path = raw_path
        self.anchor_map = defaultdict(lambda:defaultdict(int)) #{anchor {entity: count}}

        self.anchor_map_file = anchor_map_file
        self.entity_page_file = entity_page_file
        self.eprior_file = eprior_file
        self.meprior_file = meprior_file
        self.title_id_file = title_id_file
        self.redirect_file = redirect_file
        self.wiki_dump = wiki_dump
        self.log_file = log_file


        # self.f_anchor_map = codecs.open(self.anchor_map_file, "w+", encoding="utf-8")
        self.f_entity_page = codecs.open(self.entity_page_file, "w+", encoding="utf-8")
        self.f_eprior = codecs.open(self.eprior_file, "w+", encoding="utf-8") #entity prior
        self.f_meprior = codecs.open(self.meprior_file, "w+", encoding="utf-8") #entity prior given mention
        self.f_log = codecs.open(self.log_file, "w+", encoding="utf-8")
        #get title id first
        if not load_title_id:
            self.f_title_id = codecs.open(self.title_id_file, "w+", encoding="utf-8")
            self.get_title_id()
            self.save_title_id()
        else:
            self.f_title_id = codecs.open(self.title_id_file, "r", encoding="utf-8")
            self.load_title_id()

        if load_redirect_map:
            self.f_redirect = codecs.open(self.redirect_file, "r", encoding="utf-8")
            self.load_redirect_map()
        else:
            self.f_redirect = codecs.open(self.redirect_file, "w+", encoding="utf-8")
            self.get_save_redirect_map()

        self.anchor_num_sum = 0
        self.entity_num_map = defaultdict(int)

    # from url get the entity's wikipedia title
    def parse_url(self, url):
        title = url.split("/")[-1]
        title = unquote(title)
        title = title.replace("_", " ")
        return title

    def parse_json(self, content):
        content = json.loads(content)
        entity_title = self.parse_url(content["url"])
        entity_id = str(content["id"][0])
        # add title
        self.anchor_map[entity_title][entity_id] += 1
        self.entity_num_map[entity_id] += 1
        # self.title_id_map[entity_title] = entity_id

        dscpt = content["text"]
        contained_entities = set() # entities in the same page
        for anchor_info in content["annotations"]:
            anchor = anchor_info["surface_form"]
            st = anchor_info["offset"]
            ed = st + len(anchor)
            assert dscpt[st:ed] == anchor
            linked_entity = unquote(anchor_info["uri"]).replace("_", " ")
            if linked_entity in self.title_id_map:
                linked_entity_id = self.title_id_map[linked_entity]
            else:
                if linked_entity in self.redirect_map:
                    linked_entity = self.redirect_map[linked_entity]
                linked_entity_id = self.title_id_map.get(linked_entity, "-1")
            if linked_entity_id != "-1": #if it doesn't have wikipedia id, skip
                self.anchor_num_sum += 1
                self.entity_num_map[linked_entity_id] += 1
                self.anchor_map[anchor][linked_entity_id] += 1
                contained_entities.add(linked_entity_id)
            else:
                self.f_log.write(entity_title + "," + linked_entity + "\n")
        #write to the file
        contained_entities = list(contained_entities)
        # print([entity_id, entity_title, " || ".join(contained_entities)])
        entity_page_str = " ||| ".join([entity_id, entity_title, " || ".join(contained_entities)]) + "\n"
        self.f_entity_page.write(entity_page_str)

    # write the map to the file and calculate probability
    def process_anchor_map(self):
        # add redirect map to the original map
        for k, v in self.redirect_map.items():
            if v in self.title_id_map:
                eid = self.title_id_map[v]
                self.anchor_map[k][eid] += 1
                self.entity_num_map[eid] += 1

        print("[INFO] number of unique anchors: {:d}".format(len(self.anchor_map)))
        print("[INFO] number of entities that have anchor text: {:d}".format(len(self.entity_num_map)))
        print("[INFO] number of anchors: {:d}".format(self.anchor_num_sum))

        #save entity prior
        for eid, count in self.entity_num_map.items():
            # eid = self.title_id_map[etitle]
            # self.f_eprior.write("{:s} ||| {:.8f}\n".format(eid, float(count)/self.anchor_num_sum))
            self.f_eprior.write("{:s} ||| {:d}\n".format(eid, int(count)))
        self.f_eprior.close()

        #save anchor map and entity prior given mention
        for anchor, entity_count in self.anchor_map.items():
            # cur_anchor_sum = sum(list(entity_count.values()))
            # meprior = [k + " | " + str(float(v)/cur_anchor_sum) for k, v in entity_count.items()]
            meprior = [k + " | " + str(v) for k, v in entity_count.items()]
            meprior = " || ".join(meprior)
            self.f_meprior.write("{:s} ||| {:s}\n".format(anchor, meprior))

            # linked_entities = [k for k in entity_count.keys()]
            # linked_entities = " || ".join(linked_entities)
            # self.f_anchor_map.write("{:s} ||| {:s}\n".format(anchor, linked_entities))
        self.f_meprior.close()
        # self.f_anchor_map.close()

    def get_title_id(self):
        self.title_id_map = defaultdict(str)
        for sub_path in os.listdir(self.raw_path):
            print(sub_path)
            for wiki_file in os.listdir(os.path.join(self.raw_path, sub_path)):
                with  bz2.BZ2File(os.path.join(self.raw_path, sub_path, wiki_file), 'rb') as f:
                    with io.TextIOWrapper(f, encoding="utf-8") as elements:
                        for element in elements:
                            content = json.loads(element)
                            entity_title = self.parse_url(content["url"])
                            entity_id = str(content["id"][0])
                            self.title_id_map[entity_title] = entity_id

    def save_title_id(self):
        for title, id in self.title_id_map.items():
            self.f_title_id.write("{:s} ||| {:s}\n".format(title, id))
        self.f_title_id.close()
        print("[INFO] save title id, number of entities: {:d}".format(len(self.title_id_map)))

    def load_title_id(self):
        self.title_id_map = defaultdict(str)
        for line in self.f_title_id:
            tokens = line.strip().split(" ||| ")
            if len(tokens) == 2:
                self.title_id_map[tokens[0]] = tokens[1]
                self.title_id_map[tokens[0]] = tokens[1]
        print("[INFO] load title id map! {:d}".format(len(self.title_id_map)))

    def load_redirect_map(self):
        self.redirect_map = defaultdict(str)
        for line in self.f_redirect:
            tokens = line.strip().split(" ||| ")
            if len(tokens) == 2:
                self.redirect_map[tokens[1]] = tokens[0]
        print("[INFO] number of redirect title:{:d}".format(len(self.redirect_map)))
        self.f_redirect.close()

    def get_save_redirect_map(self):
        title_pattern = r'\<title\>(.*?)<\/title\>'
        redirect_pattern = r'\<redirect title\=\"(.*?)\"(.*?)\/\>'
        self.redirect_map = defaultdict(str) 
        title = ""
        with bz2.BZ2File(self.wiki_dump, 'rb') as f:
            with io.TextIOWrapper(f, encoding="utf-8") as lines:
                num = 0
                for line in lines:
                    num += 1
                    if num % 100000 == 0:
                        print("[INFO] processed {:d} lines".format(num))

                    title_match = re.search(title_pattern, line)
                    if title_match:
                        title = title_match.group(1)
                    redirect_match = re.search(redirect_pattern, line)
                    if redirect_match:
                        red = html.unescape(redirect_match.group(1)).replace("&amp;","&")
                        self.f_redirect.write(red + " ||| " + title + "\n")
                        self.redirect_map[title] = red
        self.f_redirect.close()


    def main(self):
        for sub_path in os.listdir(self.raw_path):
            print(sub_path)
            for wiki_file in os.listdir(os.path.join(self.raw_path, sub_path)):
                with  bz2.BZ2File(os.path.join(self.raw_path, sub_path, wiki_file), 'rb') as f:
                    with io.TextIOWrapper(f, encoding="utf-8") as elements:
                        for element in elements:
                            self.parse_json(element)
        self.f_entity_page.close()
        self.f_log.close()
        self.process_anchor_map()

def normalize_smooth_eprior(fname):
    map = dict()
    tot = 0.0
    smooth_tot = 0.0
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) == 2:
                map[tks[0]] = int(tks[1])
                tot += int(tks[1])
                smooth_tot += math.pow(float(tks[1]), 0.75)
    print("[INFO] tot: {:2f}".format(tot))
    print("[INFO] smooth tot: {:4f}".format(smooth_tot))
    with open(fname + "_normalize", "w+", encoding="utf-8") as f:
        for k in map:
            p = map[k] / tot
            f.write("{} ||| {:.2E}\n".format(k, Decimal(p)))

    with open(fname + "_smooth", "w+", encoding="utf-8") as f:
        for k in map:
            p = math.pow(map[k], 0.75) / smooth_tot
            f.write("{} ||| {:.2E}\n".format(k, Decimal(p)))


def normalize_meprior(fname):
    map = defaultdict(lambda:defaultdict(float))
    with open(fname, "r", encoding="utf-8") as fin:
        for line in fin:
            tks = line.strip().split(" ||| ")
            tot = 0
            if len(tks) == 2:
                mention = tks[0]
                entity_info = tks[1].split(" || ")
                tot = sum([float(x.split(" | ")[1]) for x in entity_info])
                for einfo in entity_info:
                    e, info = einfo.split(" | ")
                    map[mention][e] = float(info)/tot
    with open(fname + "_normalize", "w+", encoding="utf-8") as fout1:
        with open(fname + "_normalize_format2", "w+", encoding="utf-8") as fout2:
            for k, v in map.items():
                entity_info = []
                for kk, vv in v.items():
                    entity_info.append(" | ".join([kk, str(vv)]))
                entity_info = " || ".join(entity_info)
                fout1.write("{} ||| {}\n".format(k, entity_info))
                fout2.write("{} ||| {} ||| {}\n".format(k, " || ".join(v.keys()),  " || ".join([str(x) for x in v.values()])))

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

def de_duplicate(fname, fsave):
    with open(fname, "r", encoding="utf-8") as fin:
        sent_set = set()
        with open(fsave, "w+", encoding="utf-8") as fout:
            for line in fin:
                sent_set.add(line)
            for sent in sent_set:
                fout.write(sent)

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

def generate_pbel_data(me_file, new_me_file, link_file, en_title_id_file, train_file_path = None, train_file = None):
    '''
    :param me_file: mention ||| entities in source language ||| frequency
    :param new_me_file: mention ||| entities in English ||| frequency
    :param link_file: en wikipedia ID ||| en str ||| source language str
    :param en_title_id_file: en wikipedia ID ||| en str ||| misc. This is the filtered KB, only with name entities!
    :param train_file_path:
    :param train_file:
    :return:
    '''
    other_en_map = {}
    with open(link_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 3:
                continue
            _, en_str, other_str = tks
            other_en_map[other_str] = en_str
    print("[INFO] link map: {:d}".format(len(other_en_map)))

    en_title_id_map = {}
    with open(en_title_id_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 3:
                continue
            en_title_id_map[tks[1]] = tks[0]
    print("[INFO] en title id map: {:d}".format(len(en_title_id_map)))

    me_map = {}
    with open(me_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            mention, entities, frequency = tks
            entities = entities.split(" || ")
            frequency = frequency.split(" || ")
            duplicate_entity = []
            for e, f in zip(entities, frequency):
                duplicate_entity += [e for _ in range(int(f))]
            me_map[mention] = duplicate_entity
    print("[INFO] mention entity map: {:d}".format(len(me_map)))

    all_pairs = []
    # mention in source language ||| a list of entity in English
    # en wikipedia id ||| entity in en wikipedia || mention in source language
    with open(new_me_file, "w+", encoding="utf-8") as f:
        with open(new_me_file + "_split", "w+", encoding="utf-8") as fspl:
            for m, el in me_map.items():
                en_el = [other_en_map.get(x, "<unk>") for x in el]
                en_el = [x for x in en_el if x != "<unk>"]
                en_id_el = [en_title_id_map.get(x, "<unk>") for x in en_el]
                assert len(en_el) == len(en_id_el)

                # filter entities that do not have entity-id map in PRUNED KB file
                pruned_en_el = [x for i, x in enumerate(en_el) if en_id_el[i] != "<unk>"]
                if len(pruned_en_el) != 0:
                    f.write("{} ||| {}\n".format(m, " || ".join(list(set(pruned_en_el)))))

                for e, eid in zip(en_el, en_id_el):
                    if eid != "<unk>":
                        fspl.write("{} ||| {} ||| {}\n".format(eid, e, m))
                        all_pairs.append("{} ||| {} ||| {}\n".format(eid, e, m))

    # generate train dev test files
    if train_file:
        random.shuffle(all_pairs)
        sample_num = len(all_pairs)
        train = all_pairs[: int(sample_num * 0.7)]
        dev = all_pairs[int(sample_num * 0.7): int(sample_num * 0.9)]
        test = all_pairs[int(sample_num * 0.9): ]
        for prefix, data in zip(["train", "val", "test"], [train, dev, test]):
            with open(os.path.join(train_file_path, "med_" + prefix + "_" + train_file), "w+", encoding="utf-8") as f:
                for d in data:
                    f.write(d)
            de_duplicate(os.path.join(train_file_path, "med_" + prefix + "_" + train_file),
                         os.path.join(train_file_path, "mend_" + prefix + "_" + train_file))

        unique_test([os.path.join(train_file_path, "mend_train_" + train_file),
                     os.path.join(train_file_path, "ee_train_" + train_file)],
                    os.path.join(train_file_path, "mend_val_" + train_file),
                    os.path.join(train_file_path, "unique_mend_ee_val_" + train_file))

        unique_test([os.path.join(train_file_path, "mend_train_" + train_file),
                     os.path.join(train_file_path, "ee_train_" + train_file)],
                    os.path.join(train_file_path, "mend_test_" + train_file),
                    os.path.join(train_file_path, "unique_mend_ee_test_" + train_file))
# post processing
# save mention entity STRING file
def save_mention_entity_string(me_file, new_me_file, title_id_file):
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
            if len(tks) != 3:
                continue
            mention, entities, _ = tks
            me_map[mention] = entities.split(" || ")
    print("[INFO] mention entity map: {:d}".format(len(me_map)))

    with open(new_me_file, "w+", encoding="utf-8") as f:
        for m, el in me_map.items():
            el_str = [id_title_map.get(x, "<unk>") for x in el]
            el_str = [x for x in el_str if x != "<unk>"]
            f.write("{} ||| {}\n".format(m, " || ".join(el_str)))


# redirect entities to english and save train dev test
def redirect_to_en_str(me_file, new_me_file, link_file, en_title_id_file, train_file_path = None, train_file = None):
    other_en_map = {}
    with open(link_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 3:
                continue
            _, en_str, other_str = tks
            other_en_map[other_str] = en_str
    print("[INFO] link map: {:d}".format(len(other_en_map)))

    en_title_id_map = {}
    with open(en_title_id_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            en_title_id_map[tks[0]] = tks[1]
    print("[INFO] en title id map: {:d}".format(len(en_title_id_map)))

    me_map = {}
    with open(me_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            mention, entities = tks
            entities = entities.split(" || ")
            me_map[mention] = entities
    print("[INFO] mention entity map: {:d}".format(len(me_map)))

    all_pairs = []
    with open(new_me_file, "w+", encoding="utf-8") as f:
        with open(new_me_file + "_split", "w+", encoding="utf-8") as fspl:
            for m, el in me_map.items():
                en_el = [other_en_map.get(x, "<unk>") for x in el]
                en_el = [x for x in en_el if x != "<unk>"]
                en_id_el = [en_title_id_map.get(x, "<unk>") for x in en_el]
                f.write("{} ||| {}\n".format(m, " || ".join(en_el)))
                assert len(en_el) == len(en_id_el)
                for e, eid in zip(en_el, en_id_el):
                    if eid != "<unk>":
                        fspl.write("{} ||| {} ||| {}\n".format(eid, e, m))
                        all_pairs.append("{} ||| {} ||| {}\n".format(eid, e, m))

    if train_file:
        random.shuffle(all_pairs)
        sample_num = len(all_pairs)
        train = all_pairs[: int(sample_num * 0.7)]
        dev = all_pairs[int(sample_num * 0.7): int(sample_num * 0.9)]
        test = all_pairs[int(sample_num * 0.9): ]
        for prefix, data in zip(["train", "val", "test"], [train, dev, test]):
            with open(os.path.join(train_file_path, "me_" + prefix + "_" + train_file), "w+", encoding="utf-8") as f:
                for d in data:
                    f.write(d)



if __name__ == "__main__":
    lang = sys.argv[1]
    date = sys.argv[2]
    random.seed(1234)
    json_processor = JsonProcessor("/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_extracted".format(lang),
                                   "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/anchor_map".format(lang),
                                   "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/entity_page".format(lang),
                                   "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/eprior".format(lang),
                                   "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/meprior".format(lang),
                                   "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/log".format(lang),
                                    load_redirect_map=False,
                                    redirect_file="/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/redirect_bytitle".format(lang),
                                    wiki_dump="/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/dump_file/{}wiki/{}wiki-{}-pages-articles.xml.bz2".format(lang, lang, date),
                                    load_title_id=False,
                                    title_id_file="/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/title_id_map".format(lang),
                                   )
    json_processor.main()
    normalize_smooth_eprior("/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/eprior".format(lang))
    normalize_meprior("/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/meprior".format(lang))

    me_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/meprior".format(lang)
    me_file_with_str = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/meprior_string".format(lang)
    me_file_with_en = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/meprior_en_string".format(lang)
    title_id_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{}_results_byid/title_id_map".format(lang)
    en_link_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{}_links".format(lang)
    train_file_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/"
    en_title_id_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/kb_split/en_kb"
    train_file = "en-{}_links".format(lang)
    save_mention_entity_string_frequency(me_file, me_file_with_str, title_id_file)
    generate_pbel_data(me_file_with_str, me_file_with_en, en_link_file, en_title_id_file, train_file_path, train_file)











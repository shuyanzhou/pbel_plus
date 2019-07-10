import json
import os
import bz2
import io
import re
import math
from decimal import Decimal
from collections import defaultdict
from urllib.parse import unquote
import html
import argparse
import shutil

def parse_url(url):
    title = url.split("/")[-1]
    title = unquote(title)
    title = title.replace("_", " ")
    return title


class WikiProcessor():
    def __init__(self, raw_path, save_path, wiki_dump=None):
        self.raw_path = raw_path
        self.wiki_dump = wiki_dump

        self.anchor_map_file = os.path.join(save_path, "anchor_map")
        self.entity_page_file = os.path.join(save_path, "entity_page")
        self.entity_desc_file = os.path.join(save_path, "entity_desc")
        self.eprior_file = os.path.join(save_path, "eprior")
        self.meprior_file = os.path.join(save_path, "meprior")
        self.title_id_file = os.path.join(save_path, "title_id_map")
        self.redirect_file = os.path.join(save_path, "title_redirect_map")

        # init variables
        self.anchor_map = defaultdict(lambda: defaultdict(int))  # {anchor {entity: count}}
        self.entity_freq_counter = defaultdict(int)
        self.anchor_tot = 0

        # extract title_id_map
        if not os.path.exists(self.title_id_file):
            self.get_save_title_id_map()
        else:
            self.load_title_id_map()

        # load redirct map as some entity titles have been redirected to new titles
        if not os.path.exists(self.redirect_file):
            self.get_save_redirect_map()
        else:
            self.load_redirect_map()

    def get_save_title_id_map(self):
        self.title_id_map = defaultdict(str)
        for sub_path in os.listdir(self.raw_path):
            print(sub_path)
            for wiki_file in os.listdir(os.path.join(self.raw_path, sub_path)):
                with  bz2.BZ2File(os.path.join(self.raw_path, sub_path, wiki_file), 'rb') as f:
                    with io.TextIOWrapper(f, encoding="utf-8") as elements:
                        for element in elements:
                            content = json.loads(element)
                            entity_title = parse_url(content["url"])
                            entity_id = str(content["id"][0])
                            self.title_id_map[entity_title] = entity_id
        print("[INFO] get title id map")

        with open(self.title_id_file, "w+", encoding="utf-8") as f:
            for title, id in self.title_id_map.items():
                f.write("{:s} ||| {:s}\n".format(title, id))
            print("[INFO] save title id, number of entities: {:d}".format(len(self.title_id_map)))
        print("[INFO] save title id map")

    def load_title_id_map(self):
        self.title_id_map = defaultdict(str)
        with open(self.title_id_file, "r", encoding="utf") as f:
            for line in f:
                tokens = line.strip().split(" ||| ")
                if len(tokens) == 2:
                    self.title_id_map[tokens[0]] = tokens[1]
                    self.title_id_map[tokens[0]] = tokens[1]
        print("[INFO] load title id map! {:d}".format(len(self.title_id_map)))

    def get_save_redirect_map(self):
        title_pattern = r'\<title\>(.*?)<\/title\>'
        redirect_pattern = r'\<redirect title\=\"(.*?)\"(.*?)\/\>'
        self.redirect_map = defaultdict(str)
        title = ""
        with bz2.BZ2File(self.wiki_dump, 'rb') as f, open(self.redirect_file, "w+", encoding="utf-8") as fout:
            with io.TextIOWrapper(f, encoding="utf-8") as lines:
                num = 0
                for line in lines:
                    num += 1
                    # search title, it is always ahead of redirect pattern
                    title_match = re.search(title_pattern, line)
                    if title_match:
                        title = title_match.group(1)
                    redirect_match = re.search(redirect_pattern, line)
                    if redirect_match:
                        redirect_name = html.unescape(redirect_match.group(1)).replace("&amp;", "&")
                        fout.write(redirect_name + " ||| " + title + "\n")
                        self.redirect_map[title] = redirect_name

    def load_redirect_map(self):
        self.redirect_map = defaultdict(str)
        with open(self.redirect_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split(" ||| ")
                if len(tokens) == 2:
                    self.redirect_map[tokens[1]] = tokens[0]
            print("[INFO] number of redirect title:{:d}".format(len(self.redirect_map)))

    def parse_json(self, content, fpage, fdesc):
        content = json.loads(content)
        entity_title = parse_url(content["url"])
        entity_id = str(content["id"][0])

        # add title
        self.anchor_map[entity_title][entity_id] += 1
        self.entity_freq_counter[entity_id] += 1

        desc = content["text"]
        contained_entities = set()  # entities in the same page
        for anchor_info in content["annotations"]:
            anchor = anchor_info["surface_form"]
            st = anchor_info["offset"]
            ed = st + len(anchor)
            assert desc[st:ed] == anchor
            linked_entity = unquote(anchor_info["uri"]).replace("_", " ")

            if linked_entity in self.title_id_map:
                linked_entity_id = self.title_id_map[linked_entity]
            else:
                if linked_entity in self.redirect_map:
                    linked_entity = self.redirect_map[linked_entity]
                linked_entity_id = self.title_id_map.get(linked_entity, "-1")

            if linked_entity_id != "-1":  # record result only when the linked entity has entity id
                self.anchor_tot += 1
                self.entity_freq_counter[linked_entity_id] += 1
                self.anchor_map[anchor][linked_entity_id] += 1
                contained_entities.add(linked_entity_id)

        # write to the file
        contained_entities = list(contained_entities)
        # print([entity_id, entity_title, " || ".join(contained_entities)])
        entity_page_str = " ||| ".join([entity_id, entity_title, " || ".join(contained_entities)]) + "\n"
        fpage.write(entity_page_str)
        desc = desc.replace("\n\n", " ").replace("\n", " ")
        fdesc.write(f"{entity_title} ||| {entity_id} ||| {desc}\n")

    def save_anchor_map(self):
        # add redirect map to the original map
        for k, v in self.redirect_map.items():
            if v in self.title_id_map:
                eid = self.title_id_map[v]
                self.anchor_map[k][eid] += 1
                self.entity_freq_counter[eid] += 1

        print("[INFO] number of unique anchors: {:d}".format(len(self.anchor_map)))
        print("[INFO] number of entities that have anchor text: {:d}".format(len(self.entity_freq_counter)))
        print("[INFO] number of anchors: {:d}".format(self.anchor_tot))

        with open(self.eprior_file, "w+", encoding="utf-8") as f:
            # save entity prior
            for eid, count in self.entity_freq_counter.items():
                f.write("{:s} ||| {:d}\n".format(eid, int(count)))

        with open(self.meprior_file, "w+", encoding="utf-8") as f:
            # save anchor map and entity prior given mention
            for anchor, entity_count in self.anchor_map.items():
                meprior = [k + " | " + str(v) for k, v in entity_count.items()]
                meprior = " || ".join(meprior)
                f.write("{:s} ||| {:s}\n".format(anchor, meprior))

    def main(self):
        fpage = open(self.entity_page_file, "w+", encoding="utf-8")
        fdesc = open(self.entity_desc_file, "w+", encoding="utf-8")
        for sub_path in os.listdir(self.raw_path):
            print(sub_path)
            for wiki_file in os.listdir(os.path.join(self.raw_path, sub_path)):
                with  bz2.BZ2File(os.path.join(self.raw_path, sub_path, wiki_file), 'rb') as f:
                    with io.TextIOWrapper(f, encoding="utf-8") as elements:
                        for element in elements:
                            self.parse_json(element, fpage, fdesc)

        fpage.close()
        fdesc.close()

        self.save_anchor_map()


# normalize entity prior with |entity|, smooth by gamma = 0.75
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


# normalize mention entity prior by the occurrence of the mention
def normalize_meprior(fname):
    map = defaultdict(lambda: defaultdict(float))
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
                    map[mention][e] = float(info) / tot
    with open(fname + "_normalize", "w+", encoding="utf-8") as fout1:
        with open(fname + "_normalize_format2", "w+", encoding="utf-8") as fout2:
            for k, v in map.items():
                entity_info = []
                for kk, vv in v.items():
                    entity_info.append(" | ".join([kk, str(vv)]))
                entity_info = " || ".join(entity_info)
                fout1.write("{} ||| {}\n".format(k, entity_info))
                fout2.write(
                    "{} ||| {} ||| {}\n".format(k, " || ".join(v.keys()), " || ".join([str(x) for x in v.values()])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang")
    parser.add_argument("--date")
    parser.add_argument("-overwrite", action='store_true')
    args, _ = parser.parse_known_args()
    lang = args.lang
    date = args.date
    raw_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{lang}_extracted"
    save_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{lang}_processed"

    if args.overwrite:
        shutil.rmtree(save_path)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    wiki_dump = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/dump_file/{lang}wiki/{lang}wiki-{date}-pages-articles.xml.bz2"
    wiki_processor = WikiProcessor(raw_path, save_path, wiki_dump)
    wiki_processor.main()


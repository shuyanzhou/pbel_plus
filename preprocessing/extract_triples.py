'''
This script is used to extract information from Hiro's processed files
'''
import os
from pathlib import Path
from collections import defaultdict
import sys
from html import unescape


def lang_unit():
    return  {
            "wiki_title": "",
            "aka": set()
            }

def get_title_id_map(fname):
    title_id_map = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            title_id_map[unescape(tks[0])] = tks[1]
    print(f"[INFO] there are {len(title_id_map)} in the title id map")
    return title_id_map

def get_wikidata_wikipedia_map(fname):
    qid_title_map = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) != 2:
                continue
            qid_title_map[tks[0]] = tks[1]
    print(f"[INFO] there are {len(qid_title_map)} in the title id map")
    return qid_title_map

def extract_result(fname, result_dir, title_id_map, qid_title_map):
    tgt_lang = ["en"]
    lang_dict = defaultdict(lambda: defaultdict(lang_unit))
    all_qid = {}
    find_qid = {}
    no_wiki_id_qid = {}
    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split("\t")
            qid, wiki_title, aka, lang = tks[:4]
            if lang not in tgt_lang:
                continue
            all_qid[qid] = 1
            if qid in qid_title_map:
                find_qid[qid] = 1
                wiki_title = qid_title_map[qid]
                wiki_id = title_id_map.get(wiki_title, "NAN")
                if wiki_id ==  "NAN":
                    no_wiki_id_qid[qid] = 1
                lang_dict[lang][qid]["aka"] |= set(aka.split("||"))
                lang_dict[lang][qid]["wiki_title"] = unescape(wiki_title)
                lang_dict[lang][qid]["wiki_id"] = wiki_id
            # print(aka, set(aka.split("||")))
            # print(lang_dict[lang][qid]["aka"])

    print(f"[INFO] there are {len(all_qid)} wikidata items, find {len(find_qid)} mappings, {len(no_wiki_id_qid)} of them does not have Wikipedia ID")

    result_dir = Path(result_dir)
    for lang, contain in lang_dict.items():
        alias_file = result_dir /  f"alias.{lang}"

        with open(alias_file, "w+", encoding="utf-8") as faka:
            for qid, qid_contain in contain.items():
                wiki_title = qid_contain["wiki_title"]
                aka = list(qid_contain["aka"])
                wiki_id = qid_contain["wiki_id"]
                faka.write("{} ||| {} ||| {} ||| {}\n".format(qid, wiki_id, wiki_title, " || ".join(aka)))

    print("[INFO] done!")

if __name__ == "__main__":
    fname = "/projects/tir2/users/shuyanzh/lorelei_data/wikidata/extracted/items.txt"
    result_dir = "/projects/tir2/users/shuyanzh/lorelei_data/wikidata/lang_map"
    title_id_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/en_general/title_id_map"
    wikidata_wikipedia_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikidata/lang_map/wikidata_map.enwiki"
    title_id_map = get_title_id_map(title_id_file)
    qid_title_map = get_wikidata_wikipedia_map(wikidata_wikipedia_file)
    extract_result(fname, result_dir, title_id_map, qid_title_map)
from pathlib import Path
import os
from collections import defaultdict
import re

def extract_map(lang:list, file_name, save_folder):
    lang_map = defaultdict(lambda: defaultdict(str))
    with open(file_name, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            find = re.findall("(?<=\()(?:'[^']*'|[,\s]+|\d+)+(?=\))", line)
            for f in find:
                tks = f.split(",")
                if len(tks) != 4:
                    continue
                _, qid, wiki_lang, title = tks
                wiki_lang = wiki_lang[1:-1]
                title = title[1:-1]
                title = title.replace("\\", "")
                if wiki_lang not in lang:
                    continue
                lang_map[wiki_lang][f"Q{qid}"] = title
            print(idx)



    save_folder = Path(save_folder)
    for l, contain in lang_map.items():
        print(f"[INFO] there are {len(contain)} items in {l}")
        save_file = save_folder / f"wikidata_map.{l}"
        with open(save_file, "w+", encoding="utf-8") as f:
            for qid, title in contain.items():
                f.write("{} ||| {}\n".format(qid, title))



if __name__ == "__main__":
    lang = ["en"]
    lang = [x + "wiki" for x in lang]

    extract_map(lang,
                "/projects/tir2/users/shuyanzh/lorelei_data/wikidata/wikidatawiki-latest-wb_items_per_site.sql",
                "/projects/tir2/users/shuyanzh/lorelei_data/wikidata/lang_map")
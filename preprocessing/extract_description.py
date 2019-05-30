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

class DescriptionExtractor():
    def __init__(self, raw_path, save_file, lang_link):
        self.raw_path = raw_path
        self.f_entity_page = open(save_file, "w+", encoding="utf-8")

        self.target_wiki = []
        with open(lang_link, "r", encoding="utf-8") as f:
            for line in f:
                id = line.strip().split(" ||| ")[0]
                self.target_wiki.append(id)
        print(f"[INFO] there are {len(self.target_wiki)} target entities")

        self.find = 0

    def parse_url(self, url):
        title = url.split("/")[-1]
        title = unquote(title)
        title = title.replace("_", " ")
        return title

    def parse_json(self, content):
        content = json.loads(content)
        entity_title = self.parse_url(content["url"])
        entity_id = str(content["id"][0])
        if entity_id not in self.target_wiki:
            return
        self.find += 1
        if self.find % 10 == 0:
            print(f"[INFO] find {self.find} entities!")

        dscpt = content["text"]
        dscpt = " ".join(dscpt.splitlines())
        entity_page_str = " ||| ".join([entity_title, dscpt]) + "\n"
        self.f_entity_page.write(entity_page_str)


    def main(self):
        for sub_path in os.listdir(self.raw_path):
            print(sub_path)
            for wiki_file in os.listdir(os.path.join(self.raw_path, sub_path)):
                with  bz2.BZ2File(os.path.join(self.raw_path, sub_path, wiki_file), 'rb') as f:
                    with io.TextIOWrapper(f, encoding="utf-8") as elements:
                        for element in elements:
                            self.parse_json(element)
        self.f_entity_page.close()

if __name__ == "__main__":
    lang = "en"
    target_lang = "tpi"
    raw_path = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/{lang}_extracted/"
    save_file = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/gn_results/{lang}.txt"
    lang_link = f"/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/gn_results/en-{target_lang}_links"
    extractor = DescriptionExtractor(raw_path, save_file, lang_link)
    extractor.main()
import epitran
import os
import sys
# get analysis file

def to_ipa(fname, lang1, lang2, fsave=None):
    epitran_map = {"hi": "hin-Deva",
                   "am": "amh-Ethi",
                   "th": "tha-Thai",
                   "tr": "tur-Latn",
                   "ta": "tam-Taml",
                   "id": "ind-Latn",
                   "mr": "mar-Deva",
                   "en": "eng-Latn",
                   "ti": "tir-Ethi",
                   "te": "tel-Telu",
                   "lo": "lao-Laoo",
                   "om": "orm-Latn",
                   "kw": "kin-Latn",
                   "si": "sin-Sinh",
                   "rn": "run-Latn",
                   "so": "som-Latn"}
    if lang1 is not None:
        epi1 = epitran.Epitran(epitran_map[lang1])
    epi2 = epitran.Epitran(epitran_map[lang2])
    if fsave is None:
        fsave = fname + ".ipa"
    with open(fname, "r", encoding="utf-8") as f:
        with open(fsave, "w+", encoding="utf-8") as fout:
            for line in f:
                tks = line.strip().split(" ||| ")
                _tks = [x for x in tks]
                if lang1 is not None:
                    tks[1] = epi1.transliterate(tks[1])
                tks[2] = epi2.transliterate(tks[2])
                # tks = tks[:3] + _tks[2:3] + tks[3:]
                fout.write(" ||| ".join(tks) + "\n")


def alia_to_ipa(fname):
    epitran_map = {"hi": "hin-Deva",
                   "am": "amh-Ethi",
                   "th": "tha-Thai",
                   "tr": "tur-Latn",
                   "ta": "tam-Taml",
                   "id": "ind-Latn",
                   "mr": "mar-Deva",
                   "en": "eng-Latn",
                   "ti": "tir-Ethi",
                   "te": "tel-Telu",
                   "lo": "lao-Laoo"}
    epi1 = epitran.Epitran(epitran_map["en"])
    with open(fname, "r", encoding="utf-8") as f, open(fname + ".ipa", "w+", encoding="utf-8") as fout:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) == 4:
                aka = tks[3].split(" || ")
                aka = [epi1.transliterate(x) for x in aka]
                tks[3] = " || ".join(aka)
            fout.write(" ||| ".join(tks) + "\n")




base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"

lang = sys.argv[1]
for prefix in ["ee", "ee-me", "me"]:
    for part in ["train", "val", "test"]:
        cur_fname = os.path.join(base_path, f"{prefix}_{part}_en-{lang}_links")
        if os.path.exists(cur_fname):
            if part in ["train", "val"]:
                to_ipa(cur_fname, "en", lang)
            else:
                to_ipa(cur_fname, None, lang)

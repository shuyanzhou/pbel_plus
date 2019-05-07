import epitran
import os
# get analysis file

def to_ipa(fname, lang1, lang2):
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
    if lang1 is not None:
        epi1 = epitran.Epitran(epitran_map[lang1])
    epi2 = epitran.Epitran(epitran_map[lang2])
    with open(fname, "r", encoding="utf-8") as f:
        with open(fname + ".ipa", "w+", encoding="utf-8") as fout:
            for line in f:
                tks = line.strip().split(" ||| ")
                if lang1 is not None:
                    tks[1] = epi1.transliterate(tks[1])
                tks[2] = epi2.transliterate(tks[2])
                fout.write(" ||| ".join(tks) + "\n")



base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
fnames = ["ee_train_en-{}_links", "ee_mend_train_en-{}_links", "ee_mend_train_en-{}_links.mid", 
          "unique_mend_ee_val_en-{}_links", "unique_mend_ee_val_en-{}_links.mid"]
'''
for lang in ["am", "hi", "th", "ta"]:
    for fname in fnames:
        cur_fname = os.path.join(base_path, fname.format(lang))
        if ".mid" in cur_fname:
            to_ipa(cur_fname, lang, lang)
            print(cur_fname, lang, lang)
        else:
            to_ipa(cur_fname, "en", lang)
            print(cur_fname, "en", lang)

fnames = ["ee_test_en-{}_links", "unique_mend_ee_test_en-{}_links"]
for lang in ["ti", "mr", "lo", "te"]:
    for fname in fnames:
        cur_fname = os.path.join(base_path, fname.format(lang))
        to_ipa(cur_fname, None, lang)
'''
# for lang in ["am", "hi", "th", "ta"]:
#     cur_fname = os.path.join(base_path, "en-{}_links".format(lang))
#     to_ipa(cur_fname, None, lang)

for lang in ["am", "hi", "th", "ta", "id", "tr"]:
    cur_fname = os.path.join(base_path, "ee_val_en-{}_links".format(lang))
    to_ipa(cur_fname, "en", lang)
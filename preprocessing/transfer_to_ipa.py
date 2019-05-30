import epitran
import os
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
                   "lo": "lao-Laoo"}
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
                tks = tks[:3] + _tks[2:3] + tks[3:]
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

# for lang in ["am", "hi", "th", "ta", "id", "tr"]:
#     cur_fname = os.path.join(base_path, "ee_val_en-{}_links".format(lang))
#     to_ipa(cur_fname, "en", lang)


lang = "hi"
file_name = os.path.join(base_path, "ee_mend_train_en-{}_links".format(lang))
fsave = os.path.join(base_path, "../ipa", "ee_mend_train_en-{}_links.ipa".format(lang))
to_ipa(file_name, None, lang, fsave)

lang = "mr"
file_name = os.path.join(base_path, "unique_mend_ee_test_en-{}_links".format(lang))
fsave = os.path.join(base_path, "../ipa", "unique_mend_ee_test_en-{}_links.ipa".format(lang))
to_ipa(file_name, None, lang, fsave)
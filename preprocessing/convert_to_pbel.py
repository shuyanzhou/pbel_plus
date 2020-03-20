import os
import sys
import epitran

def convert_mention_file(ner_file, saved_file, lang, to_ipa):
    lang_map = {
        "il5": "tir-Ethi",
        "il6": "orm-Latn",
        "il9": "kin-Latn",
        "il10": "sin-Sinh",
        "il11": "ori-Orya",
        "il12": "ilo-Latn"
    }

    print("[INFO] converting NER file")
    with open(ner_file, "r", encoding="utf-8") as fin, open(saved_file, "w+", encoding="utf-8") as fout:
        tot = 0
        for line in fin:
            tot += 1
            tks = line.strip()
            mention = tks
            fout.write(f"-1 ||| -1 ||| {mention} ||| -1\n")

        print(f"[INFO] write {tot} lines to new file")

    if to_ipa:
        epi = epitran.Epitran(lang_map[lang])
        with open(saved_file, "r", encoding="utf-8") as fin, open(saved_file + ".ipa", "w+", encoding="utf-8") as fout:
            for line in fin:
                tks = line.strip().split(" ||| ")
                tks[2] = epi.transliterate(tks[2])
                fout.write(" ||| ".join(tks) + "\n")

if __name__ == "__main__":
    folder = sys.argv[1] # data folder
    file_prefix = sys.argv[2]
    lang = sys.argv[3]
    to_ipa = True if int(sys.argv[4]) == 1 else False
    assert not to_ipa
    ner_file = os.path.join(folder, file_prefix + ".ner")
    save_file = os.path.join(folder, file_prefix)
    convert_mention_file(ner_file, save_file, lang, to_ipa)




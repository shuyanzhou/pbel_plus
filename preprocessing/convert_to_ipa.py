import epitran
import argparse

map_file = "/home/shuyanzh/workshop/lor_edl/map.tsv"
def to_ipa(fname, lang1, lang2):
    epitran_map = {}
    with open(map_file, "r", encoding="utf-8") as f:
        for line in f:
            tks = line.strip().split("\t")
            epitran_map[tks[1]] = tks[0]
    epi1 = epitran.Epitran(epitran_map[lang1])
    epi2 = epitran.Epitran(epitran_map[lang2])
    fsave = fname + ".ipa"
    with open(fname, "r", encoding="utf-8") as f, open(fsave, "w+", encoding="utf-8") as fout:
        for line in f:
            tks = line.strip().split(" ||| ")
            if len(tks) < 3:
                continue
            _tks = [x for x in tks]
            tks[1] = epi1.transliterate(tks[1])
            tks[2] = epi2.transliterate(tks[2])
            # tks = tks[:3] + _tks[2:3] + tks[3:]
            fout.write(" ||| ".join(tks) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang1", default="en")
    parser.add_argument("--lang2", required=True)
    parser.add_argument("--fname", required=True)

    args,  _ = parser.parse_known_args()
    to_ipa(args.fname, args.lang1, args.lang2)

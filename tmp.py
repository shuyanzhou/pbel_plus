with open("ngrams", "r", encoding="utf-8") as fin, open("ngrams_new", "w+", encoding="utf-8") as fout:
	for line in fin:
		tks = line.strip()
		fout.write(f"-1 ||| -1 ||| {tks} ||| -1\n")

import epitran
import os

ipa_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/sample_data/ee-me_train_en-hi_links.ipa"
all_data = []
with open(ipa_file, "r", encoding="utf-8") as f:
    for line in f:
        all_data.append(line.strip())


graph_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/ee-me_train_en-hi_links"
graph_data = []
with open(graph_file, "r", encoding="utf-8") as f:
    epi1 = epitran.Epitran("eng-Latn")
    epi2 = epitran.Epitran("hin-Deva")
    for line in f:
        tks = line.strip().split(" ||| ")
        tks[1] = epi1.transliterate(tks[1])
        tks[2] = epi2.transliterate(tks[2])
        if " ||| ".join(tks) in all_data:
            graph_data.append(line)

graph_data = set(graph_data)
print(len(graph_data))

save_file = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/sample_data/ee-me_train_en-hi_links"
with open(save_file, "w+", encoding="utf-8") as f:
    for d in graph_data:
        f.write(d)
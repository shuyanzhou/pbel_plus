import os
import sys
# fname = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/hi_results_byid/meprior_string"
#
# tot = 0
# exact_match = 0
# with open(fname, "r", encoding="utf-8") as f:
#     for line in f:
#         tks = line.strip().split(" ||| ")
#         if len(tks) == 3:
#             mention, entities = tks[0], tks[1].split(" || ")
#             tot += len(entities)
#             for e in entities:
#                 if e == mention:
#                     exact_match += 1
#
# print(exact_match, tot, exact_match / float(tot))

lang = sys.argv[1]
data = sys.argv[2]
lang_link_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{}_links".format(lang)
fname = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data/{}_en-{}_links".format(data, lang)

lang_link_map = {}
with open(lang_link_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        if len(tks) == 3:
            lang_link_map[tks[1]] = tks[2]
print(len(lang_link_map))


tot = 0
exact_match = 0
error = 0
with open(fname, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        tot += 1
        if tks[1] not in lang_link_map:
            error += 1
            continue
        if tks[2] == lang_link_map[tks[1]]:
            exact_match += 1
print(error, exact_match, tot, exact_match / float(tot))
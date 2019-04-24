import os
import sys

lang = sys.argv[1]
plang = sys.argv[2]

base_path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
train_file = "ee_train_en-{}_links".format(lang)
train_file = os.path.join(base_path, train_file)

test_file = "unique_mend_ee_test_en-{}_links".format(plang)
test_file = os.path.join(base_path, test_file)

test_file_ee = "ee_test_en-{}_links".format(plang)
test_file_ee = os.path.join(base_path, test_file_ee)

# check train test mention/entity overlap
train_mention, train_entity = [], []
test_mention, test_entity = [], []

with open(train_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        train_mention.append(tks[2])
        train_entity.append(tks[0])

with open(test_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        test_mention.append(tks[2])
        test_entity.append(tks[0])

tot = len(test_mention)
occur_mention = 0
occur_entity = 0
for m, e in zip(test_mention, test_entity):
    if m in train_mention:
        occur_mention += 1
    if e in train_entity:
        occur_entity += 1

print(occur_mention, occur_entity, tot, occur_mention/tot, occur_entity/tot)


# check pivoting exact match

lang_link_file = "/projects/tir2/users/shuyanzh/lorelei_data/wikipedia/links/en-{}_links".format(lang)
lang_link_map = {}
with open(lang_link_file, "r", encoding="utf-8") as f:
    for line in f:
        tks = line.strip().split(" ||| ")
        if len(tks) == 3:
            lang_link_map[tks[0]] = tks[2]
print(len(lang_link_map))


for tf in [test_file, test_file_ee]:
    tot = 0
    exact_match = 0
    with open(tf, "r", encoding="utf-8") as f:
        for line in f:
            tot += 1
            tks = line.strip().split(" ||| ")
            eid, _, mention = tks[:3]
            estr = lang_link_map.get(eid, "")
            if estr == mention:
                exact_match += 1

    print(exact_match, tot, exact_match/tot)
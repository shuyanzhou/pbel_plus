import os
import sys

# two letters
lang = sys.argv[1]
# train or test
data = sys.argv[2]
if data == "train":
    fname = "train_template.sh"
elif data == "test":
    fname = "test_template.sh"
elif data == "ptest":
    test_lang = sys.argv[3]
    fname = "test_template2.sh"
else:
    raise NotImplementedError

with open(fname, "r", encoding="utf-8") as f:
    template = f.read()

if data != "ptest":
    flang = lang
else:
    flang = test_lang

with open(data + "_" + flang + ".sh", "w+", encoding="utf-8") as f:
    if data == "ptest":
        t = template.replace("plang", test_lang)
    else:
        t = template
    t = t.replace("en-lang", "en-{}".format(lang))
    f.write(t)

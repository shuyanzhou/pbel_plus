import os
import sys

# two letters
lang = sys.argv[1]
# train or test
data = sys.argv[2]
if data == "train_ipa":
    fname = "train_template3.sh"
elif data == "train_grapheme":
    fname = "train_template4.sh"
elif data == "test":
    fname = "test_template.sh"
elif data == "ptest_ipa":
    test_lang = sys.argv[3]
    fname = "test_template3.sh"
elif data == "ptest_grapheme":
    test_lang = sys.argv[3]
    fname = "test_template4.sh"
else:
    raise NotImplementedError

with open(fname, "r", encoding="utf-8") as f:
    template = f.read()

if "ptest" not in data:
    flang = lang
else:
    flang = test_lang

with open(data + "_" + flang + ".sh", "w+", encoding="utf-8") as f:
    if "ptest" in data:
        t = template.replace("plang", test_lang)
    else:
        t = template
    t = t.replace("en-lang", "en-{}".format(lang))
    f.write(t)

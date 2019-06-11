'''
this script is used to rename all model/result files and generate new bash file template
me_val/test_en-lang_links is non-duplicated data, they are also unique compare to the mend_train and ee_train
'''

import os
import sys

def change_name(path, old_name, new_name):
    os.rename(os.path.join(path, old_name), os.path.join(path, new_name))

# path = "/projects/tir2/users/shuyanzh/lorelei_data/pbel/data"
# for fname in os.listdir(path):
#     if "unique_mend_ee" in fname:
#         change_name(path, fname, fname.replace("unique_mend_ee_",  "me_"))
#
#     if "ee_mend_train" in fname:
#         change_name(path, fname, fname.replace("ee_mend_train_", "ee-me_train_"))


# all_path = ["/projects/tir2/users/shuyanzh/lorelei_data/pbel/models", "/projects/tir2/users/shuyanzh/lorelei_data/pbel/c2i_maps"]
# for path in all_path:
#     for fname in os.listdir(path):
        # new_name = fname.replace("ee_mend", "ee-me").replace("grapheme", "graph")
        # if "charagram" in new_name:
        #     new_name = new_name.replace("charagram_cosine-hinge", "char-cosine-hinge")
        # else:
        #     new_name = new_name.replace("cosine-hinge", "lstm-cosine-hinge")
        # change_name(path, fname, new_name)
        # if "char" in fname:
        #     new_name = fname.replace("lstm-", "")
        #     change_name(path, fname, new_name)

all_path = ["/projects/tir2/users/shuyanzh/lorelei_data/pbel/results",
            "/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/split",
            "/projects/tir2/users/shuyanzh/lorelei_data/pbel/results/analysis",
            "/projects/tir2/users/shuyanzh/lorelei_data/pbel/models",
            "/projects/tir2/users/shuyanzh/lorelei_data/pbel/c2i_maps"][-2:]

for path in all_path:
    for fname in os.listdir(path):
        if os.path.isdir(fname):
            continue
        # new_name = fname.replace("ee_mend", "ee-me").replace("grapheme", "graph")
        # if "unique_mend_ee_" in fname:
        #     new_name = fname.replace("unique_mend_ee_", "me_")
        # if "_ee_mend_" in new_name:
        #     new_name = new_name.replace("_ee_mend_", "_ee-me_")
        # if "charagram" in new_name:
        #     new_name = new_name.replace("charagram_cosine-hinge", "char-cosine-hinge")
        # else:
        #     new_name = new_name.replace("cosine-hinge", "lstm-cosine-hinge")
        #
        if "char" in fname:
            new_name = fname.replace("lstm-", "")
        else:
            new_name = fname.replace("lstm-lstm-lstm-lstm", "lstm")
        if "aka" in new_name:
            new_name = new_name.replace("graph_aka", "aka_graph")
            new_name = new_name.replace("ipa_aka", "aka_ipa")
        if "-multi" in new_name:
            new_name = new_name.replace("-multi", "_multi")
        if "ee_test" in new_name:
            new_name = new_name.replace("ee_test", "ee")
        change_name(path, fname, new_name)

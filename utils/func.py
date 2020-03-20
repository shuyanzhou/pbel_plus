import numpy as np
from typing import List

def list2nparr(all_lists:List[List], hidden_size:int, merge=False):
    combined_encodings = []
    for version_list in all_lists:
        # last batch might not match the size
        encodings = np.concatenate(version_list, axis=0)
        combined_encodings.append(encodings)
    if merge:
        return np.vstack(combined_encodings)
    else:
        return combined_encodings


def append_multiple_encodings(encoding_list, encodings, encoding_num):
    diff_encodings = np.vsplit(encodings, encoding_num)
    for idx, e in enumerate(diff_encodings):
        encoding_list[idx].append(e)


class FileInfo:
    def __init__(self):
        self.src_file_name = None
        self.src_str_idx = None
        self.src_id_idx = None
        self.trg_file_name = None
        self.trg_str_idx = None
        self.trg_id_idx = None
        self.mid_file_name = None
        self.mid_str_idx = None
        self.mid_id_idx = None

    def set_all(self, file_name, src_str_idx, trg_str_idx, id_idx, type_idx):
        self.src_file_name = file_name
        self.trg_file_name = file_name
        self.src_str_idx = int(src_str_idx)
        self.trg_str_idx = int(trg_str_idx)
        self.src_id_idx = int(id_idx)
        self.trg_id_idx = int(id_idx)
        self.trg_type_idx = int(type_idx)

    def set_src(self, file_name, str_idx, id_idx):
        self.src_file_name = file_name
        self.src_str_idx = int(str_idx)
        self.src_id_idx = int(id_idx)

    def set_trg(self, file_name, str_idx, id_idx, type_idx):
        self.trg_file_name = file_name
        self.trg_str_idx = int(str_idx)
        self.trg_id_idx = int(id_idx)
        self.trg_type_idx = int(type_idx)

    def set_mid(self, file_name, str_idx, id_idx, type_idx):
        self.mid_file_name = file_name
        self.mid_str_idx = int(str_idx)
        self.mid_id_idx = int(id_idx)
        self.mid_type_idx = int(type_idx)
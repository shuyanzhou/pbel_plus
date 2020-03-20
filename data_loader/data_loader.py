class BaseDataLoader:
    def __init__(self, is_train, args,
                 train_file: FileInfo, dev_file: FileInfo, test_file: FileInfo):
        self.batch_size = args.batch_size
        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.map_file = args.map_file
        self.pad_str = "<UNK>"
        self.pad_idx = 0
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.use_mid = args.use_mid
        self.trg_encoding_num = args.trg_encoding_num
        self.mid_encoding_num = args.mid_encoding_num
        self.load_alia_map(args.alia_file)
        self.n_gram_threshold = args.n_gram_threshold
        self.max_position = 0
        if is_train:
            self.init_train()
        else:
            self.init_test()

    def n_gram_filter(self, data, freq_map):
        filter_data = []
        for cur_data in data:
            all_string_idx = cur_data[0][0]
            cur_filter_string, cur_filter_st, cur_filter_ed = [], [], []
            for cur_version, cur_version_string in enumerate(all_string_idx):
                filter_idx = []
                for idx, ngram_idx in enumerate(cur_version_string):
                    if ngram_idx == self.pad_idx or freq_map[idx] < self.n_gram_threshold:
                        continue
                    else:
                        filter_idx.append(idx)
                if len(filter_idx) != 0:
                    filter_string = [cur_version_string[x] for x in filter_idx]
                    # if self.position_embedding:
                    #     st_idx = cur_data[0][1][cur_version]
                    #     ed_idx = cur_data[0][2][cur_version]
                    #     filter_st = [st_idx[x] for x in filter_idx]
                    #     filter_ed = [ed_idx[x] for x in filter_idx]
                else:
                    filter_string = [self.pad_idx]
                    # if self.position_embedding:
                    #     filter_st = [0]
                    #     filter_ed = [0]

                cur_filter_string.append(filter_string)
                # if self.position_embedding:
                #     cur_filter_ed.append(filter_ed)
                #     cur_filter_st.append(filter_st)

            # if self.position_embedding:
            #     all_info = [cur_filter_string, cur_filter_st, cur_filter_ed]
            # else:
            all_info = [cur_filter_string]
            filter_data.append([all_info, cur_data[1]])

        return filter_data

    def init_train(self):
        self.x2i_src = defaultdict(lambda: len(self.x2i_src))
        self.x2i_trg = defaultdict(lambda: len(self.x2i_trg))
        # make sure pad is 0
        self.x2i_src[self.pad_str]
        self.x2i_trg[self.pad_str]
        self.src_freq_map = Counter()
        self.trg_freq_map = Counter()
        self.train_src = list(self.load_data(self.train_file.src_file_name, self.train_file.src_str_idx,
                                             self.train_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
        self.train_trg = list(self.load_data(self.train_file.trg_file_name, self.train_file.trg_str_idx,
                                             self.train_file.trg_id_idx, is_src=False,
                                             encoding_num=self.trg_encoding_num,
                                             type_idx=self.train_file.trg_type_idx))
        # save map
        self.save_map(self.x2i_src, self.map_file + "_src.pkl")
        self.save_map(self.x2i_trg, self.map_file + "_trg.pkl")
        self.save_map(self.src_freq_map, self.map_file + "_src_freq.pkl")
        self.save_map(self.trg_freq_map, self.map_file + "_trg_freq.pkl")

        if self.use_mid:
            # if self.share_vocab:
            #     self.x2i_mid = self.x2i_src
            #     self.mid_freq_map = self.src_freq_map
            # else:
            self.x2i_mid = defaultdict(lambda: len(self.x2i_mid))
            self.x2i_mid[self.pad_str]
            self.mid_freq_map = Counter()
            self.train_mid = list(
                self.load_data(self.train_file.mid_file_name, self.train_file.mid_str_idx, self.train_file.mid_id_idx,
                               is_src=False, is_mid=True, encoding_num=self.mid_encoding_num,
                               type_idx=self.train_file.mid_type_idx,
                               auto_encoding=self.mid_auto_encoding))
            self.save_map(self.x2i_mid, self.map_file + "_mid.pkl")
            self.save_map(self.mid_freq_map, self.map_file + "_mid_freq.pkl")
            self.mid_vocab_size = len(self.x2i_mid)
            self.x2i_mid = defaultdict(lambda: self.x2i_mid[self.pad_str], self.x2i_mid)
            self.mid_freq_map = defaultdict(lambda: float('-inf'), self.mid_freq_map)
        else:
            self.train_mid = None
            self.mid_vocab_size = 0

        # self.non_neg_mask = self.get_non_negative_mask()

        # sort training data by input length
        self.src_vocab_size = len(self.x2i_src)
        self.trg_vocab_size = len(self.x2i_trg)
        self.x2i_src = defaultdict(lambda: self.x2i_src[self.pad_str], self.x2i_src)
        self.x2i_trg = defaultdict(lambda: self.x2i_trg[self.pad_str], self.x2i_trg)
        self.src_freq_map = defaultdict(lambda: float('-inf'), self.src_freq_map)
        self.trg_freq_map = defaultdict(lambda: float('-inf'), self.trg_freq_map)

        if self.dev_file:
            self.dev_src = list(self.load_data(self.dev_file.src_file_name, self.dev_file.src_str_idx,
                                               self.dev_file.src_id_idx, is_src=True, encoding_num=1, type_idx=None))
            self.dev_trg = list(self.load_data(self.dev_file.trg_file_name, self.dev_file.trg_str_idx,
                                               self.dev_file.trg_id_idx, is_src=False,
                                               encoding_num=self.trg_encoding_num, type_idx=self.dev_file.trg_type_idx))
            n = min(len(self.dev_src), 2000)
            self.dev_src, self.dev_trg = self.dev_src[:n], self.dev_trg[:n]
            if self.use_mid:
                self.dev_mid = list(self.load_data(self.dev_file.mid_file_name, self.dev_file.mid_str_idx,
                                                   self.dev_file.mid_id_idx, is_src=False, is_mid=True,
                                                   encoding_num=self.mid_encoding_num,
                                                   type_idx=self.dev_file.mid_type_idx))
                self.dev_mid = self.dev_mid[:n]
            else:
                self.dev_mid = None
        else:
            self.dev_src, self.dev_trg, self.dev_mid = None, None, None

        if self.n_gram_threshold != 0:
            self.train_src = self.n_gram_filter(self.train_src, self.src_freq_map)
            self.train_trg = self.n_gram_filter(self.train_trg, self.trg_freq_map)
            if self.use_mid:
                self.train_mid = self.n_gram_filter(self.train_mid, self.mid_freq_map)

            # recover from training frequency
            if self.dev_file:
                self.dev_src = self.n_gram_filter(self.dev_src, self.src_freq_map)
                self.dev_trg = self.n_gram_filter(self.dev_trg, self.trg_freq_map)
                if self.use_mid:
                    self.dev_mid = self.n_gram_filter(self.dev_mid, self.mid_freq_map)

    # def get_non_negative_mask(self):
    #     id_idx_map = defaultdict(list)
    #     for idx, (_, kb_id) in enumerate(self.train_src):
    #         id_idx_map[kb_id].append(idx)
    #     mask = torch.zeros((len(self.train_src), len(self.train_src)))
    #     for _, idx_list in id_idx_map.items():
    #         idx_pairs = combinations(idx_list, 2)
    #         for i, j in idx_pairs:
    #             mask[i, j] = 1
    #             mask[j, i] = 1
    #     mask += torch.eye(len(self.train_src), len(self.train_src))
    #     mask = mask.long()
    #     # mask = mask.to(device)
    #     return mask

    def init_test(self):
        self.x2i_src = self.load_map(self.map_file + "_src.pkl")
        self.x2i_trg = self.load_map(self.map_file + "_trg.pkl")
        self.src_freq_map = self.load_map(self.map_file + "_src_freq.pkl", float('-inf')) if os.path.exists(
            self.map_file + "_src_freq.pkl") \
            else defaultdict(int)
        self.trg_freq_map = self.load_map(self.map_file + "_trg_freq.pkl", float('-inf')) if os.path.exists(
            self.map_file + "_trg_freq.pkl") \
            else defaultdict(int)
        if self.use_mid:
            self.x2i_mid = self.load_map(self.map_file + "_mid.pkl")
            self.mid_freq_map = self.load_map(self.map_file + "_mid_freq.pkl", float('-inf'))
        else:
            self.x2i_mid = None
        self.i2c_src = {v: k for k, v in self.x2i_src.items()}
        self.i2c_trg = {v: k for k, v in self.x2i_trg.items()}
        if self.test_file.src_file_name is not None:
            self.test_src = list(self.load_data(self.test_file.src_file_name,
                                                self.test_file.src_str_idx, self.test_file.src_id_idx,
                                                is_src=True, encoding_num=1, type_idx=None))
        if self.test_file.trg_file_name is not None:
            self.test_trg = list(self.load_data(self.test_file.trg_file_name,
                                                self.test_file.trg_str_idx, self.test_file.trg_id_idx,
                                                is_src=False, encoding_num=self.trg_encoding_num,
                                                type_idx=self.test_file.trg_type_idx))
        if self.test_file.mid_file_name is not None:
            self.test_mid = list(self.load_data(self.test_file.mid_file_name,
                                                self.test_file.mid_str_idx, self.test_file.mid_id_idx, is_src=False,
                                                is_mid=True,
                                                encoding_num=self.mid_encoding_num,
                                                type_idx=self.test_file.mid_type_idx))

        if self.n_gram_threshold != 0:
            if self.test_file.src_file_name is not None:
                self.test_src = self.n_gram_filter(self.test_src, self.src_freq_map, True)
            if self.test_file.trg_file_name is not None:
                self.test_trg = self.n_gram_filter(self.test_trg, self.trg_freq_map, True)
            if self.test_file.mid_file_name is not None:
                self.test_mid = self.n_gram_filter(self.test_mid, self.mid_freq_map, True)

    def load_alia_map(self, fname):
        if fname != "HOLDER":
            self.title_alia_map = defaultdict(list)
            self.id_alia_map = defaultdict(list)
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    tks = line.strip().split(" ||| ")
                    if len(tks) != 4:
                        continue
                    aka = tks[3].split(" || ")
                    self.title_alia_map[tks[2]] = aka
                    if tks[1] != "NAN":
                        self.id_alia_map[tks[1]] = aka
            print(f"[INFO] there are {len(self.title_alia_map)} / {len(self.id_alia_map)} items in aka")
        else:
            print("[WARNING] no alia file found!")

    def get_alias(self, tks, str_idx, id_idx, encoding_num):
        id = tks[id_idx]
        title = tks[str_idx]
        alias = self.title_alia_map.get(title, []) + self.id_alia_map.get(id, [])
        alias = [x for x in alias if x != title]

        if len(alias) < encoding_num:
            alias = [title for x in range(encoding_num - len(alias))] + alias
        # randomly select encoding num - 1 alias
        else:
            selected_idx = np.random.choice(len(alias), encoding_num - 1, replace=False)
            alias = [alias[x] for x in selected_idx]
            alias = [title] + alias

        assert len(alias) == encoding_num

        return alias

    def load_all_data(self, file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx):
        pass

    def load_data(self, file_name, str_idx, id_idx, is_src, encoding_num, type_idx, is_mid=False):
        if is_src:
            x2i_map = self.x2i_src
            freq_map = self.src_freq_map
        else:
            x2i_map = self.x2i_trg
            freq_map = self.trg_freq_map
        if is_mid:
            x2i_map = self.x2i_mid
            freq_map = self.mid_freq_map
        return self.load_all_data(file_name, str_idx, id_idx, x2i_map, freq_map, encoding_num, type_idx)

    def transform_one_batch(self, *args, **kwargs) -> list:
        pass

    def new_batch(self) -> BaseBatch:
        pass

    def extract_idx(self, encoding_num, idx_list):
        all_idx = []
        for i in range(encoding_num):
            all_idx += [idx_list[idx][i] for idx in range(len(idx_list))]
        assert len(all_idx) == len(idx_list) * encoding_num

        return all_idx

    # data from one side
    def prepare_batch(self, side_data, data_idx, encoding_num):
        # this is a list of words
        words = [side_data[idx][0][0] for idx in data_idx]
        # expand words to list
        all_words = self.extract_idx(encoding_num, words)
        word_idx_tensor, *other_info = self.transform_one_batch(all_words)

        # if self.position_embedding:
        #     st = [side_data[idx][0][1] for idx in data_idx]
        #     all_st = self.extract_idx(encoding_num, st)
        #     ed = [side_data[idx][0][2] for idx in data_idx]
        #     all_ed = self.extract_idx(encoding_num, ed)
        #     st_idx_tensor = self.transform_one_batch(all_st)[0]
        #     ed_idx_tensor = self.transform_one_batch(all_ed)[0]
        #     merge_tensor = torch.stack((word_idx_tensor, st_idx_tensor, ed_idx_tensor), dim=-1)
        # else:
        merge_tensor = word_idx_tensor

        kb_ids = [side_data[idx][1] for idx in data_idx]
        batch_info = [merge_tensor, *other_info]

        return batch_info, kb_ids

    def create_batch(self, dataset, data_src=None, data_trg=None, data_mega=None, data_mid=None) -> List[BaseBatch]:
        batches = []
        non_none = [x for x in [data_src, data_trg, data_mid] if x is not None][0]
        data_idx = [i for i in range(len(non_none))]
        if dataset == "train":
            random.shuffle(data_idx)
        for i in range(0, len(data_idx), self.batch_size):
            batch = self.new_batch()
            cur_size = min(self.batch_size, len(data_idx) - i)
            cur_data_idx = data_idx[i:i + cur_size]
            if data_src is not None:
                batch_info, src_gold_kb_ids = self.prepare_batch(data_src, cur_data_idx, encoding_num=1)
                batch.set_src(*batch_info, src_gold_kb_ids)
            if data_trg is not None:
                batch_info, trg_kb_ids = self.prepare_batch(data_trg, cur_data_idx, encoding_num=self.trg_encoding_num)
                batch.set_trg(*batch_info, trg_kb_ids)
            if data_mega is not None:
                # TODO this needs to be fix with multiple encodings!
                batch_info, trg_kb_ids = self.prepare_batch(data_mega, cur_data_idx, encoding_num=self.trg_encoding_num)
                batch.set_mega(*batch_info, trg_kb_ids)
            if data_mid is not None:
                batch_info, mid_kb_ids = self.prepare_batch(data_mid, cur_data_idx, encoding_num=self.mid_encoding_num)
                batch.set_mid(*batch_info, mid_kb_ids)
            # move to device
            batch.to(device)
            batches.append(batch)

        return batches

    # pad both source and target words
    def create_batches(self, dataset: str, is_src=None, is_mid=None) -> List[BaseBatch]:
        # self.train_mid could be None!
        # training time
        if dataset == "train":
            batches = self.create_batch(dataset, self.train_src, self.train_trg, data_mid=self.train_mid)
        elif dataset == "dev":
            batches = self.create_batch(dataset, self.dev_src, self.dev_trg, data_mid=self.dev_mid)
        # test time, load data separately
        else:
            assert is_src is not None and is_mid is not None
            if is_mid:
                batches = self.create_batch(dataset, data_src=None, data_trg=None, data_mega=None,
                                            data_mid=self.test_mid)
            else:
                if is_src:
                    batches = self.create_batch(dataset, self.test_src, None, None, None)
                else:
                    batches = self.create_batch(dataset, None, self.test_trg, None, None)

        return batches

    # def create_megabatch(self, model: Encoder):
    #     # only for training
    #     data_src, data_trg = self.train_src, self.train_trg
    #     data_idx = [i for i in range(len(data_src))]
    #     random.shuffle(data_idx)
    #     for i in range(0, len(data_idx), self.mega_batch_size):
    #         batch = self.new_batch()
    #         cur_size = min(self.mega_batch_size, len(data_idx) - i)
    #         cur_data_idx = data_idx[i: i + cur_size]
    #         # src
    #         batch_info, src_gold_kb_ids = self.prepare_batch(data_src, cur_data_idx)
    #         batch.set_src(*batch_info, src_gold_kb_ids)
    #         # trg
    #         batch_info, trg_kb_ids = self.prepare_batch(data_trg, cur_data_idx)
    #         batch.set_trg(*batch_info, trg_kb_ids)
    #
    #         with torch.no_grad():
    #             model.eval()
    #             batch.to(device)
    #             M, _ = model.calc_batch_similarity(batch, use_negative=False, use_mid=False, proportion=0,
    #                                                trg_encoding_num=0, mid_encoding_num=0)
    #             model.train()
    #             raise NotImplementedError
    #
    #         negative_num = min(1, cur_size - 1)
    #
    #         # mask the non negative samples and the diagonal
    #         idx_tensor = torch.LongTensor(cur_data_idx)
    #         non_neg_mask = torch.index_select(self.non_neg_mask, 0, idx_tensor)
    #         non_neg_mask = torch.index_select(non_neg_mask, 1, idx_tensor)
    #         non_neg_mask = non_neg_mask.to(device)
    #         masked_M = M.masked_fill(non_neg_mask == 1, -1e9)
    #
    #         # negative_idx = [batch_size, 1]
    #         _, negative_idx = torch.topk(masked_M, k=negative_num, dim=-1)
    #         # negative sample
    #         cur_negative_idx = [cur_data_idx[idx.item()] for idx in negative_idx]
    #         mega_src = [data_src[idx] for idx in cur_data_idx]
    #         mega_trg = [data_trg[idx] for idx in cur_data_idx]
    #         mega_negative = [data_trg[idx] for idx in cur_negative_idx]
    #
    #         cur_mega_batch = self.create_batch("train", mega_src, mega_trg, mega_negative)
    #         for b in cur_mega_batch:
    #             yield b

    def save_map(self, map, map_file):
        # save map
        with open(map_file, "wb") as f:
            pickle.dump(dict(map), f)
            print("[INFO] save x to idx map to :{}, len: {:d}".format(map_file, len(map)))

    def load_map(self, map_file, default_return=None):
        with open(map_file, "rb") as f:
            m = pickle.load(f)
            if default_return is None:
                default_return = m[self.pad_str]
            m = defaultdict(lambda: default_return, m)
            print("[INFO] load x to idx map from {}, len: {:d}".format(map_file, len(m)))
            return m
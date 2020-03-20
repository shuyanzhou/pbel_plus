import argparse
import os
import copy
import pprint
def str2bool(s):
    if s == "0":
        return False
    else:
        return True

def argps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help='model to use for encoding strings',
                        choices=('charagram', 'charcnn', 'lstm', 'avg_lstm'),
                        type=str, default='charagram')
    parser.add_argument("--is_train", type=str2bool, default=False)
    parser.add_argument("--use_mid", help="whether to use entity in that language",
                        type=str2bool, default=False)

    # multi version
    parser.add_argument("--trg_encoding_num", type=int, default=1)
    parser.add_argument("--mid_encoding_num", type=int, default=1)
    parser.add_argument("--trg_type_idx", type=int, default=3)
    parser.add_argument("--mid_type_idx", type=int, default=3)
    parser.add_argument("--kb_type_idx", type=int, default=2)
    parser.add_argument("--pivot_type_idx", type=int, default=3)
    parser.add_argument("--alia_file", type=str, default="HOLDER")
    parser.add_argument("--method", default="base")

    # filter
    parser.add_argument("--n_gram_threshold", help="ignore n gram with less than the min frequency", type=int, default=0)

    # middle stuff
    parser.add_argument("--train_mid_file", default="")
    parser.add_argument("--dev_mid_file", default="")
    parser.add_argument("--mid_str_idx", help="HRL entity string", type=int, default=1)
    parser.add_argument("--mid_id_idx", type=int, default=0)
    parser.add_argument("--mid_proportion", help="the proportion used in the similarity matrix", type=float, default=0.3)

    # for cnn
    parser.add_argument("--pooling_method", choices=("mean", "sum", "max"))
    # train
    parser.add_argument("--train_file", default="")
    parser.add_argument("--dev_file", default="")
    parser.add_argument("--map_file", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--resume", type=str2bool, default=False)
    parser.add_argument("--src_idx", help="LRL or HRL mention string", type=int, default=2)
    parser.add_argument("--trg_idx", help="KB entity string index", type=int, default=1)
    parser.add_argument("--trg_id_idx", help="KB id index", type=int, default=0)
    parser.add_argument("--val_topk", type=int, default=30)

    # training details
    parser.add_argument("--similarity_measure", choices=("cosine", "bl", "lcosine"), required=True)
    parser.add_argument("--objective", choices=("hinge", "mle"), required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--hidden_size", help="bi-direction", type=int)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--trainer", choices=('adam', 'sgd', 'sgd_mo', 'rmsp'))
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lr_decay", type=str2bool, default=False)
    parser.add_argument("--lr_scaler", type=float)
    parser.add_argument("--max_epoch", type=int, default=200)

    # test
    parser.add_argument("--test_epoch", type=str, default="best")
    parser.add_argument("--test_file", default="")
    parser.add_argument("--test_str_idx", help="HRL or LRL string", type=int, default=2)
    parser.add_argument("--test_id_idx", help="EN wiki id", type=int, default=0)
    parser.add_argument("--encoded_test_file", default="")
    parser.add_argument("--load_encoded_test", type=str2bool, default=False)
    parser.add_argument("--record_recall", type=str2bool, default=True)
    parser.add_argument("--kb_file")
    parser.add_argument("--kb_str_idx", type=int, default=1)
    parser.add_argument("--kb_id_idx", type=int, default=0)
    parser.add_argument("--encoded_kb_file", default="")
    parser.add_argument("--load_encoded_kb", type=str2bool, default=False)
    parser.add_argument("--no_pivot_result", default="")

    #intermedia stuff
    #pivoting
    parser.add_argument("--pivot_file", default="pivot")
    parser.add_argument("--pivot_str_idx", type=int, default=2)
    parser.add_argument("--pivot_id_idx", type=int, default=0)
    parser.add_argument("--encoded_pivot_file")
    parser.add_argument("--load_encoded_pivot", type=str2bool, default=False)
    parser.add_argument("--pivot_result", default="")
    parser.add_argument("--pivot_is_src", type=str2bool)
    parser.add_argument("--pivot_is_mid", type=str2bool)


    args, _ = parser.parse_known_args()

    # convert intermediate stuff
    # name, file_name, str_idx, id_idx, encoded_file, load_encoded, is_src
    args.intermediate_stuff = []
    args.intermediate_stuff.append(["pivot", args.pivot_file, args.pivot_str_idx, args.pivot_id_idx, args.pivot_type_idx,
                                    args.encoded_pivot_file, args.load_encoded_pivot, args.pivot_is_src, args.pivot_is_mid])

    # result files
    args.result_file = {}
    args.result_file["no_pivot"] = args.no_pivot_result + ".id"
    args.result_file["no_pivot_str"] = args.no_pivot_result + ".str"
    args.result_file["pivot"] = args.pivot_result + ".id"
    args.result_file["pivot_str"] = args.pivot_result + ".str"

    # print config
    pprint.pprint(vars(args))
    return args
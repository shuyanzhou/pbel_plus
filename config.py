import argparse
import os
import copy

def str2bool(s):
    if s == "0":
        return False
    else:
        return True

def argps():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_train", type=str2bool, default=False)
    parser.add_argument("--mega", help="whether to use mega batch to find negative samples",
                        type=str2bool, default=False)
    parser.add_argument("--use_mid", help="whether to use entity in that language",
                        type=str2bool, default=False)

    # mega batch
    parser.add_argument("--mega_size", type=int, default=40)
    parser.add_argument("--method", default="base")

    # middle stuff
    parser.add_argument("--share_vocab", help="whether to share src and middle vocab and encoding model",
                        type=str2bool, default=False)
    parser.add_argument("--train_mid_file", default="")
    parser.add_argument("--dev_mid_file", default="")
    parser.add_argument("--mid_str_idx", help="HRL entity string", type=int, default=1)
    parser.add_argument("--mid_id_idx", type=int, default=0)
    parser.add_argument("--mid_proportion", help="the proportion used in the similarity matrix", type=float, default=0.3)

    # train
    parser.add_argument("--train_file", default="")
    parser.add_argument("--dev_file", default="")
    parser.add_argument("--map_file", default="")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--src_idx", help="LRL or HRL mention string", type=int, default=2)
    parser.add_argument("--trg_idx", help="KB entity string index", type=int, default=1)
    parser.add_argument("--trg_id_idx", help="KB id index", type=int, default=0)
    parser.add_argument("--use_panphon", type=str2bool, default=False)
    
    # training details
    parser.add_argument("--similarity_measure", choices=("cosine", "bl"))
    parser.add_argument("--objective", choices=("hinge", "mle"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_size", type=int, default=64)
    parser.add_argument("--hidden_size", help="bi-direction", type=int, default=1024)
    parser.add_argument("--margin", type=int, default=1)
    parser.add_argument("--trainer", choices=('adam', 'sgd'), default='adam')
    parser.add_argument("--learning_rate", type=float)
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
    parser.add_argument("--pivot_is_src", type=str2bool, default=True)
    parser.add_argument("--pivot_is_mid", type=str2bool, default=False)


    args, _ = parser.parse_known_args()

    # convert intermediate stuff
    # name, file_name, str_idx, id_idx, encoded_file, load_encoded, is_src
    args.intermediate_stuff = []
    args.intermediate_stuff.append(["pivot", args.pivot_file, args.pivot_str_idx, args.pivot_id_idx,
                                    args.encoded_pivot_file, args.load_encoded_pivot, args.pivot_is_src, args.pivot_is_mid])

    # result files
    args.result_file = {}
    args.result_file["no_pivot"] = args.no_pivot_result + ".id"
    args.result_file["no_pivot_str"] = args.no_pivot_result + ".str"
    args.result_file["pivot"] = args.pivot_result + ".id"
    args.result_file["pivot_str"] = args.pivot_result + ".str"

    # print config
    for k, v in vars(args).items():
        if v:
            print(str(k) + ":", v)
    return args
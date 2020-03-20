import sys
import functools
import torch
from torch import optim
import time
import numpy as np
from utils.similarity_calculator import Similarity
import argparse
from typing import List, Generator
from criterion import NSHingeLoss, MultiMarginLoss, CrossEntropyLoss
from utils.constant import RANDOM_SEED, PATIENT, EPOCH_CHECK, DEVICE, UPDATE_PATIENT
from models.base_encoder import Encoder
from data_loader.data_loader import BaseDataLoader, BaseBatch
from utils.func import list2nparr, append_multiple_encodings, FileInfo

print = functools.partial(print, flush=True)
device = DEVICE


def calc_batch_loss(model, criterion, batch: BaseBatch, proportion, trg_encoding_num, mid_encoding_num):
    # src_tensor, src_lens, src_perm_idx, trg_tensor, trg_kb_id, trg_lens, trg_perm_idx
    similarity, diff = model.calc_batch_similarity(batch, use_negative=True, use_mid=True,
                                                   proportion=proportion, trg_encoding_num=trg_encoding_num, mid_encoding_num=mid_encoding_num)
    if diff is not None:
        loss = criterion(similarity) + diff
    else:
        loss = criterion(similarity)
    return loss

def get_unique_kb_idx(kb_id_list: list):
    find_kb_ids = []
    unique_kb_idx = []
    for i, id in enumerate(kb_id_list):
        if id in find_kb_ids:
            continue
        else:
            unique_kb_idx.append(i)
            find_kb_ids.append(id)
    return np.array(unique_kb_idx)


def merge_encodings(encoding1:List[np.ndarray], encoding2:List[np.ndarray]):
    assert len(encoding1) == len(encoding2)
    combined_encodings = [np.vstack([x, y]) for x, y in zip(encoding1, encoding2)]
    combined_encodings = np.vstack(combined_encodings)
    return combined_encodings

# evaluate the whole dataset
def eval_data(model: Encoder, train_batches:List[BaseBatch], dev_batches: List[BaseBatch], similarity_measure: Similarity, args_dict: dict):
    use_mid = args_dict["use_mid"]
    topk = args_dict["topk"]
    trg_encoding_num = args_dict["trg_encoding_num"]
    mid_encoding_num = args_dict["mid_encoding_num"]
    # treat train target strings as the KB
    recall = 0
    tot = 0
    KB_encodings = [[] for _ in range(trg_encoding_num)]
    KB_ids = []
    for batch in train_batches:
        cur_encodings = np.array(model.calc_encode(batch, is_src=False).cpu())
        append_multiple_encodings(KB_encodings, cur_encodings, trg_encoding_num)
        KB_ids += batch.trg_kb_ids
    assert len(KB_encodings[0]) == len(train_batches)
    KB_encodings = list2nparr(KB_encodings, model.hidden_size)

    src_encodings = []
    trg_encodings = [[] for _ in range(trg_encoding_num)]
    trg_kb_ids = []
    for batch in dev_batches:
        src_encodings.append(np.array(model.calc_encode(batch, is_src=True).cpu()))
        cur_encodings = np.array(model.calc_encode(batch, is_src=False).cpu())
        append_multiple_encodings(trg_encodings, cur_encodings, trg_encoding_num)
        trg_kb_ids += batch.trg_kb_ids
    assert len(src_encodings) == len(dev_batches)
    assert len(trg_encodings[0]) == len(dev_batches)

    src_encodings = list2nparr([src_encodings], model.hidden_size, merge=True)
    trg_encodings = list2nparr(trg_encodings, model.hidden_size)


    # TODO might need it in the future
    # prune KB_encodings so that all entities are unique
    unique_kb_idx = get_unique_kb_idx(KB_ids)
    KB_encodings = [x[unique_kb_idx] for x in KB_encodings]

    all_trg_encodings = merge_encodings(trg_encodings, KB_encodings)
    n = max(all_trg_encodings.shape[0], 160000)
    all_trg_encodings = all_trg_encodings[:n]
    # calculate similarity`
    # [dev_size, dev_size + kb_size]
    scores = similarity_measure(src_encodings, all_trg_encodings, is_src_trg=True, split=True,
                                pieces=10, negative_sample=None, encoding_num=trg_encoding_num)
    encoding_scores = np.copy(scores)
    if use_mid:
        mid_KB_encodings = [[] for _ in range(mid_encoding_num)]
        for batch in train_batches:
            cur_encodings = np.array(model.calc_encode(batch, is_src=False, is_mid=True).cpu())
            append_multiple_encodings(mid_KB_encodings, cur_encodings, mid_encoding_num)
            KB_ids += batch.trg_kb_ids
        assert len(mid_KB_encodings[0]) == len(train_batches)
        mid_KB_encodings = list2nparr(mid_KB_encodings, model.hidden_size)

        mid_encodings = [[] for _ in range(mid_encoding_num)]
        for batch in dev_batches:
            cur_encodings = np.array(model.calc_encode(batch, is_src=False, is_mid=True).cpu())
            append_multiple_encodings(mid_encodings, cur_encodings, mid_encoding_num)
        assert len(mid_encodings[0]) == len(dev_batches)
        mid_encodings = list2nparr(mid_encodings, model.hidden_size)

        # TODO might need it in the future
        # mid_KB_encodings = mid_KB_encodings[unique_kb_idx]
        all_mid_encodings = merge_encodings(mid_encodings, mid_KB_encodings)
        all_mid_encodings = all_mid_encodings[:n]
        all_mid_encodings = all_mid_encodings[:n]

        mid_scores = similarity_measure(src_encodings, all_mid_encodings,
                                        is_src_trg=False, split=True, pieces=10, negative_sample=None, encoding_num=mid_encoding_num)
        scores = np.maximum(scores, mid_scores)
    for entry_idx, entry_scores in enumerate(scores):
        ranked_idxes = entry_scores.argsort()[::-1]
        # the correct index is entry_idx
        if entry_idx in ranked_idxes[:topk]:
            recall += 1
        tot += 1

    recall_2 = 0
    for entry_idx, entry_scores in enumerate(encoding_scores):
        ranked_idxes = entry_scores.argsort()[::-1]
        # the correct index is entry_idx
        if entry_idx in ranked_idxes[:topk]:
            recall_2 += 1

    return [recall, recall_2], tot

def reset_bias(module):
    param = module.state_dict()["bias_hh_l0"]
    p = torch.zeros_like(param)
    p[512:1024] = 1.0
    param.copy_(p)

    param = module.state_dict()["bias_hh_l0_reverse"]
    p = torch.zeros_like(param)
    p[512:1024] = 1.0
    param.copy_(p)

def run(data_loader: BaseDataLoader, encoder: Encoder, criterion, optimizer: optim, scheduler: optim.lr_scheduler,
          similarity_measure: Similarity, save_model,
          args:argparse.Namespace):
    encoder.to(device)
    best_accs = {"encode_acc": float('-inf'), "pivot_acc": float('-inf')}
    last_update = 0
    dev_arg_dict = {
        "use_mid": args.use_mid,
        "topk": args.val_topk,
        "trg_encoding_num": args.trg_encoding_num,
        "mid_encoding_num": args.mid_encoding_num
    }
    # lr_decay = scheduler is not None
    # if lr_decay:
    #     print("[INFO] using learning rate decay")
    for ep in range(args.max_epoch):
        encoder.train()
        train_loss = 0.0
        start_time = time.time()
        # if not args.mega:
        train_batches = data_loader.create_batches("train")
        # else:
        #     if ep <= 30:
        #         train_batches = data_loader.create_batches("train")
        #     else:
        #         train_batches = data_loader.create_megabatch(encoder)
        batch_num = 0
        t = 0
        for idx, batch in enumerate(train_batches):
            optimizer.zero_grad()
            cur_loss = calc_batch_loss(encoder, criterion, batch, args.mid_proportion, args.trg_encoding_num, args.mid_encoding_num)
            train_loss += cur_loss.item()
            cur_loss.backward()
            # optimizer.step()

            for p in list(filter(lambda p: p.grad is not None, encoder.parameters())):
                t += p.grad.data.norm(2).item()

            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=5)
            optimizer.step()

            if encoder.name == "bilstm":
                # set all but forget gate bias to 0
                reset_bias(encoder.src_lstm)
                reset_bias(encoder.trg_lstm)
                # pass
            batch_num += 1
        print("[INFO] epoch {:d}: train loss={:.8f}, time={:.2f}".format(ep, train_loss / batch_num,
                                                                         time.time()-start_time))
        # print(t)

        if (ep + 1) % EPOCH_CHECK == 0:
            with torch.no_grad():
                encoder.eval()
                # eval
                train_batches = data_loader.create_batches("train")
                dev_batches = data_loader.create_batches("dev")
                start_time = time.time()

                recall, tot = eval_data(encoder, train_batches, dev_batches, similarity_measure, dev_arg_dict)
                dev_pivot_acc = recall[0] / float(tot)
                dev_encode_acc = recall[1] / float(tot)
                if dev_encode_acc > best_accs["encode_acc"]:
                    best_accs["encode_acc"] = dev_encode_acc
                    best_accs["pivot_acc"] = dev_pivot_acc
                    last_update = ep + 1
                    save_model(encoder, ep + 1, train_loss / batch_num, optimizer, args.model_path + "_" + "best" + ".tar")
                save_model(encoder, ep + 1, train_loss / batch_num, optimizer, args.model_path + "_" + "last" + ".tar")
                print("[INFO] epoch {:d}: encoding/pivoting dev acc={:.4f}/{:.4f}, time={:.2f}".format(
                                                                                            ep, dev_encode_acc, dev_pivot_acc,
                                                                                            time.time()-start_time))
                if args.lr_decay and ep + 1 - last_update > UPDATE_PATIENT:
                    new_lr = optimizer.param_groups[0]['lr'] * args.lr_scaler
                    best_info  = torch.load(args.model_path + "_" + "best" + ".tar")
                    encoder.load_state_dict(best_info["model_state_dict"])
                    optimizer.load_state_dict(best_info["optimizer_state_dict"])
                    optimizer.param_groups[0]['lr'] = new_lr
                    print("[INFO] reload best model ..")

                if ep + 1 - last_update > PATIENT:
                    print("[FINAL] in epoch {}, the best develop encoding/pivoting accuracy = {:.4f}/{:.4f}".format(ep + 1,
                                                                                                                    best_accs["encode_acc"],
                                                                                                                    best_accs["pivot_acc"]))
                    break
        # if lr_decay:
        #     scheduler.step()

def init_train(args, DataLoader):
    train_file = FileInfo()
    train_file.set_all(args.train_file, args.src_idx, args.trg_idx, args.trg_id_idx, args.trg_type_idx)
    train_file.set_mid(args.train_mid_file, args.mid_str_idx, args.mid_id_idx, args.mid_type_idx)
    dev_file = FileInfo()
    dev_file.set_all(args.dev_file, args.src_idx, args.trg_idx, args.trg_id_idx, args.trg_type_idx)
    dev_file.set_mid(args.dev_mid_file, args.mid_str_idx, args.mid_id_idx, args.mid_type_idx)
    data_loader = DataLoader(is_train=True, args=args, train_file=train_file,
                             dev_file=dev_file, test_file=None)
    similarity_measure = Similarity(args.similarity_measure)

    if args.objective == "hinge":
        criterion = MultiMarginLoss(device, margin=args.margin, reduction="mean")
        # criterion = MultiMarginLoss(device, margin=args.margin, reduction="sum")
    elif args.objective == "mle":
        criterion = CrossEntropyLoss(device, reduction="mean")
    else:
        raise NotImplementedError

    return data_loader, criterion, similarity_measure

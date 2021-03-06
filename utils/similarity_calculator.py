import torch
import numpy as np
import time
import gc
from utils.constant import DEVICE
device = DEVICE

class Similarity:
    '''
    :param src_encoded:[batch_size, hidden_size]
    :param trg_encoded: [batch_size, hidden_size]
    :param bilinear_tensor: [hidden_size, hidden_size]
    :return: bilinear_score: [batch_size, batch_size]
    '''
    def __init__(self, method):
        self.method = method
        print("[INFO] using {} to measure similarity".format(self.method))

    def set_src_mid_bl(self, t: torch.Tensor):
        self.src_mid_bl = t

    def set_src_trg_bl(self, t: torch.Tensor):
        self.src_trg_bl = t

    def set_src_affine(self, t: torch.Tensor):
        self.src_affine = t

    def set_trg_affine(self, t: torch.Tensor):
        self.trg_affine = t

    def split_large_matrix(self, matrix:np.ndarray, pieces):
        k, m = divmod(matrix.shape[0], pieces)
        for i in range(pieces):
            cur_matrix = matrix[i * k + min(i, m):(i + 1) * k + min(i + 1, m)]
            cur_matrix = torch.from_numpy(cur_matrix).to(device).float()
            yield cur_matrix

    # def split_large_matrix(self, matrix:np.ndarray, pieces):
    #     batch_size = matrix.shape[0] // pieces
    #     tot = matrix.shape[0]
    #     for _ in range(0, tot, batch_size):
    #         cur_batch_size = min(batch_size, matrix.shape[0])
    #         cur_matrix = matrix[:cur_batch_size]
    #         cur_matrix = torch.from_numpy(cur_matrix).to(device).float()
    #         matrix = np.delete(matrix, [x for x in range(cur_batch_size)], axis=0)
    #         yield cur_matrix

    def calc_cosine_similarity(self, src_encoded, trg_encoded):
        src_norm = torch.norm(src_encoded, dim=1, keepdim=True)
        src_norm_encoded = src_encoded / src_norm
        trg_norm = torch.norm(trg_encoded, dim=1, keepdim=True)
        trg_norm_encoded = trg_encoded / trg_norm
        similarity = torch.matmul(src_norm_encoded, torch.transpose(trg_norm_encoded, 1, 0))
        return similarity

    def calc_cosine_similarity_split(self, src_encoded, trg_encoded, pieces):
        src_encoded = torch.from_numpy(src_encoded).to(device).float()
        src_norm = torch.norm(src_encoded, dim=1, keepdim=True)
        src_norm_encoded = src_encoded / src_norm
        # split target to 10 pieces
        similarity_collection = []
        for idx, cur_trg_encoded in enumerate(self.split_large_matrix(trg_encoded, pieces)):
            cur_trg_norm = torch.norm(cur_trg_encoded, dim=1, keepdim=True)
            cur_trg_norm_encoded = cur_trg_encoded / cur_trg_norm
            # [src_size, piece_size]
            similarity = torch.matmul(src_norm_encoded, torch.transpose(cur_trg_norm_encoded, 1, 0))
            similarity_collection.append(similarity.cpu().numpy())
        print("[INFO] done calculating similarity")
        # concatenate
        similarity_collection = np.hstack(tuple(similarity_collection))
        return similarity_collection
    
    def calc_linear_cosine(self, src_encoded, trg_encoded, is_src_trg):
        if is_src_trg:
            src = torch.mm(src_encoded, self.src_affine)
            trg = torch.mm(trg_encoded, self.trg_affine)
        else: # src and mid does not need transformation
            src = src_encoded
            trg = trg_encoded
        return self.calc_cosine_similarity(src, trg)

    def calc_linear_cosine_split(self, src_encoded, trg_encoded, is_src_trg, pieces):
        src_encoded = torch.from_numpy(src_encoded).to(device).float()
        similarity_collection = []
        for cur_trg_encoded in self.split_large_matrix(trg_encoded, pieces):
            similarity = self.calc_linear_cosine(src_encoded, cur_trg_encoded, is_src_trg)
            similarity_collection.append(similarity.cpu().numpy())
        similarity_collection = np.hstack(tuple(similarity_collection))
        return similarity_collection

    def calc_bilinear(self, src_encoded, trg_encoded, is_src_trg):
        if is_src_trg:
            bl_tensor = self.src_trg_bl
        else:
            bl_tensor = self.src_mid_bl
        bilinear_score = torch.mm(src_encoded, torch.mm(bl_tensor, torch.transpose(trg_encoded, 1, 0)))
        return bilinear_score

    def calc_bilinear_split(self, src_encoded, trg_encoded, is_src_trg, pieces):
        if is_src_trg:
            bl_tensor = self.src_trg_bl
        else:
            bl_tensor = self.src_mid_bl
        src_encoded = torch.from_numpy(src_encoded).to(device).float()
        # split target to 10 pieces
        # 10 [src_size, piece_size]
        similarity_collection = []
        for cur_trg_encoded in self.split_large_matrix(trg_encoded, pieces):
            similarity = torch.mm(src_encoded, torch.mm(bl_tensor, torch.transpose(cur_trg_encoded, 1, 0)))
            similarity_collection.append(similarity.cpu().numpy())
        # concatenate
        similarity_collection = np.hstack(tuple(similarity_collection))
        return similarity_collection

    def __call__(self, src_encoded:np.ndarray, trg_encoded:np.ndarray,
                 is_src_trg,
                 split, pieces, negative_sample, encoding_num):
        '''
        :param negative_sample: it is not None only during TRAINING (calc_batch_similarity with has negative_sample on)
        the resulted similarity matrix is sent to criterion for use.
        it will put the correct answer at the first column.
        for others, it returns the similarity between 2 matrixes without shifting the position
        '''
        src_encoded = src_encoded
        trg_encoded = trg_encoded
        # in training process, mm is enough, we use pytorch
        if not split:
            if self.method == "cosine":
                M = self.calc_cosine_similarity(src_encoded, trg_encoded)
            elif self.method == "bl":
                M = self.calc_bilinear(src_encoded, trg_encoded, is_src_trg)
            elif self.method == "lcosine":
                M = self.calc_linear_cosine(src_encoded, trg_encoded, is_src_trg)
            else:
                raise NotImplementedError
            if encoding_num != 1:
                mul_M = torch.chunk(M, encoding_num, dim=1)
                # concatenate, [batch_size, batch_size, encoding_num]
                con_M = torch.stack(mul_M, dim=2)
                M, _ = torch.max(con_M, dim=2)
        else: # here the returned matrix is numpy
            if self.method == "cosine":
                M =  self.calc_cosine_similarity_split(src_encoded, trg_encoded, pieces)
            elif self.method == "bl":
                M = self.calc_bilinear_split(src_encoded, trg_encoded, is_src_trg, pieces)
            elif self.method == "lcosine":
                M = self.calc_linear_cosine_split(src_encoded, trg_encoded, is_src_trg, pieces)
            else:
                raise NotImplementedError
            # find the version with highest score
            if encoding_num != 1:
                mul_M = np.hsplit(M, encoding_num)
                M = np.maximum.reduce(mul_M)

        # prune to have only some negative samples. only true when training 
        if negative_sample is not None:
            M = torch.transpose(M, 1, 0)
            if negative_sample != 0:
                negative_sample = int(negative_sample)
                assert M.shape[0] != M.shape[1] and M.shape[1] % M.shape[0] == 0
                batch_size = M.shape[0]
                pruned_M = torch.zeros((batch_size, negative_sample + 1))
                pruned_M = pruned_M.to(device)
                for i in range(negative_sample + 1):
                    score = torch.diag(M[:, i * batch_size: (i+1) * batch_size])
                    pruned_M[:, i] = score
                M = pruned_M
            else:
                # assert  M.shape[0] == M.shape[1]
                # move diag to the first element
                batch_size = M.shape[0]
                idx = [[i for i in range(batch_size)] for _ in range(batch_size)]
                for i, x in enumerate(idx):
                    x.remove(i)
                idx = torch.LongTensor(idx).to(device)
                tM = torch.gather(M, dim=1, index=idx)
                M = torch.cat((torch.diag(M).view(-1, 1), tM), dim=1)
                assert M.shape[0] == batch_size and M.shape[1] == batch_size
        return M

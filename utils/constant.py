import torch
# general
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 8756

#train
PATIENT = 50
EPOCH_CHECK = 2
UPDATE_PATIENT = 5
#model
PP_VEC_SIZE = 22


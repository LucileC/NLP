import torch

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 256
# VOC_SIZE = 500 #vocab.n_words#50000 #for both source and target
VOC_SIZE = 50000
OUTPUT_SIZE = VOC_SIZE #10 # ?? 
BASELINE_VOC_SIZE = VOC_SIZE
MAX_LENGTH = 400

path_dev = "data/msmarco_2wellformed/dev_v2.0_well_formed.json"
path_train = "data/msmarco_2wellformed/train_v2.0_well_formed.json"
path_eval = "data/msmarco_2wellformed/evalpublicwellformed.json"

SOS_token = 0
EOS_token = 1
UNK = 2

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(DEVICE)

RAPID_TEST = False
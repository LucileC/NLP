import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import prepare_dataset as prep
from constants import *


class EncoderLSTM(nn.Module):

    def __init__(self,embedding_size=EMBEDDING_SIZE,hidden_size=HIDDEN_SIZE,voc_size=VOC_SIZE):
        super(EncoderLSTM,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.voc_size = voc_size

        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.bilstm = nn.LSTM(embedding_size, hidden_size, num_layers =1, bidirectional=True)

    def initHidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_size)), # 2 because bidirectional
        Variable(torch.zeros(2, 1, self.hidden_size)))

    def forward(self, input, hidden):
        #         print(input)
        embedded = self.embedding(input).view(1,1,-1)
        output, hidden = self.bilstm(embedded, hidden)
        return hidden

def test():
    input_var, _ = prep.test(path_dev)
    encoder = EncoderLSTM()
    encoder.to(DEVICE)
    input_length = len(input_var)
    encoder_hidden = encoder.initHidden()        
    h = Variable(torch.zeros(input_length, encoder.hidden_size*2))        
    for ei in range(input_length):
        encoder_hidden = encoder(input_var[ei],encoder_hidden)
        h[ei] = torch.cat((encoder_hidden[0][0],encoder_hidden[1][0]),1)
    print(input_var.size())
    print(h.size())
    return h


if __name__ == "__main__": 
    test()
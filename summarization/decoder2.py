import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy

import prepare_dataset as prep
import encoder
from constants import *

class AttnDecoderLSTM2(nn.Module):
    
    def __init__(self,embedding_size=EMBEDDING_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE,
                 voc_size=VOC_SIZE,max_length=MAX_LENGTH):
        super(AttnDecoderLSTM2,self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.voc_size = voc_size
        
        self.embedding = nn.Embedding(voc_size, embedding_size)
        self.decoder_bilstm = nn.LSTM(self.embedding_size,self.hidden_size, num_layers=1, bidirectional = True)
        self.attn_Ws = nn.Linear(hidden_size *2, hidden_size) #?
        self.attn_Wh = nn.Linear(hidden_size *2, hidden_size) #?
        self.attn_v = nn.Linear(hidden_size, 1)
        self.lin_V1 = nn.Linear(hidden_size*4, hidden_size)
        self.lin_V2 = nn.Linear(hidden_size, output_size)
        
    def initHidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_size, device=DEVICE)), # 2 because bidirectional
                Variable(torch.zeros(2, 1, self.hidden_size, device=DEVICE)))
        
    def forward(self,input,hidden,h):
        emb = self.embedding(input).view(1,1,-1)
        output, hidden = self.decoder_bilstm(emb,hidden)
        s_t = torch.cat((hidden[0][0],hidden[1][0]),1)
        
        # attention distribution
        if CUDA:
            h = h.cuda()
            s_t = s_t.cuda()
        Wh = self.attn_Wh(h) # dim: number of words in input , 256
        Ws_t = self.attn_Ws(s_t) # dim: 1 , 256
        Wh_Ws_t_d = torch.add(Wh,Ws_t) # dim: number of words in input , 256 
        Wh_Ws_t_d = F.tanh(Wh_Ws_t_d) # dim: number of words in input , 256 
        e_t = self.attn_v(Wh_Ws_t_d) # dim: number of words in input , 1 
        e_t = torch.t(e_t) # dim: 1, number of words in input
        a_t = F.softmax(e_t, dim=1) # dim: 1, number of words in input
        hstar_t = torch.bmm(a_t.unsqueeze(0),h.unsqueeze(0)) # dim: 1, 1 , 512
        print(hstar_t)
        # vocabulary distribution
        v1 = torch.cat((s_t.unsqueeze(0),hstar_t),dim=2) # dim: 1, 1, 1024
        v1 = self.lin_V1(v1) # dim: 1, 1, 256
        v2 = self.lin_V2(v1) # dim: 1, 1, vocabulary size
        Pvocab = F.log_softmax(v2,dim=2) # dim: 1, 1, vocabulary size
        
        return Pvocab[0], hidden    
      
# decoder = AttnDecoderLSTM()
# target_length = len(target_var)
# hidden = decoder.initHidden()         
# input = Variable(torch.LongTensor([[SOS_token]]))      
# loss = 0
# criterion = nn.NLLLoss()
# for di in range(target_length):
#     output, hidden = decoder(input, hidden, h)
# #     outputs[di] = output
#     input = target_var[di]

#     loss += criterion(output,target_var[di])
# print (loss/target_length)

def test(input_target_pair):
    print('Test decoder')
    input_var, target_var = input_target_pair
    h = encoder.test(input_target_pair)
    decoder2 = AttnDecoderLSTM2()
    decoder2.to(DEVICE)
    loss = 0
    criterion = nn.NLLLoss()    

    decoder_hidden = decoder2.initHidden()         
    for ei in range(len(target_var)):
        decoder_output, decoder_hidden = decoder2(target_var[ei],decoder_hidden, h)
        loss += criterion(decoder_output,target_var[ei])
    print(decoder_output)
    print (loss/len(target_var))


if __name__ == "__main__": 
    input_target_pair = prep.test(path_dev,verbose=False)
    test(input_target_pair)
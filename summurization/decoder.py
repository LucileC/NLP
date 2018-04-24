import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import prepare_dataset as prep
import encoder
from constants import *

class AttnDecoderLSTM(nn.Module):
    
    def __init__(self,embedding_size=EMBEDDING_SIZE,hidden_size=HIDDEN_SIZE,output_size=OUTPUT_SIZE,
                 voc_size=VOC_SIZE,max_length=MAX_LENGTH):
        super(AttnDecoderLSTM,self).__init__()
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
        self.test = nn.Linear(hidden_size*2,output_size)
        
    def initHidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_size)), # 2 because bidirectional
                Variable(torch.zeros(2, 1, self.hidden_size)))
        
    def forward_rec(self,input,hidden):
        emb = self.embedding(input).view(1,1,-1)
        output, hidden = self.decoder_bilstm(emb,hidden)
        return output, hidden
    

    def forward(self, target_var, h):
        input_length = h.size()[0]
        target_length = len(target_var)
        hidden = self.initHidden()        
        outputs = Variable(torch.zeros(target_length, self.hidden_size*2))  
        s = Variable(torch.zeros(target_length,self.hidden_size*2))
        input = Variable(torch.LongTensor([[SOS_token]]))    
        
        for di in range(target_length):
            output, hidden = self.forward_rec(input, hidden)
            outputs[di] = output[0]
            s[di] = torch.cat((hidden[0][0],hidden[1][0]),1)
            input = target_var[di]
        
        # attention distribution
        Wh = self.attn_Wh(h) # dim: # of words in input , 256
        Ws = self.attn_Ws(s) # dim: # of words in target , 256
        Wh_Ws_d = Variable(torch.zeros(target_length,input_length,256)) # dim: # of words in target, # of words in input , 256 
        for i in range(target_length):
            Wh_Ws_d[i] = torch.add(Wh,Ws[0])
        Wh_Ws_d = F.tanh(Wh_Ws_d) # dim: # of words in target, # of words in input , 256 
        e = self.attn_v(Wh_Ws_d) # dim: # of words in target, # of words in input , 1 
        e = e.permute(0,2,1) # dim: # of words in target, 1, # of words in input 
        a = F.softmax(e, dim=2) # dim: # of words in target, 1, # of words in input 
        h_extended = torch.add(Variable(torch.zeros(target_length,input_length,self.hidden_size*2)),h)
        hstar = torch.bmm(a,h_extended) # dim: #of target words, 1, 512
        # vocabulary distribution
        v1 = torch.cat((s.unsqueeze(1),hstar),dim=2) # dim: #of target words, 1, 1024
        v1 = self.lin_V1(v1) # dim: #of target words, 1, 256
        v2 = self.lin_V2(v1) # dim: #of target words, 1, vocabulary size
        Pvocab = F.softmax(v2,dim=2) # dim: #of target words, 1, vocabulary size
        x = self.test(s).unsqueeze(1)
#         print(s.size())
        Pvocab = F.log_softmax(x,2)
        
        return Pvocab, outputs    
      
def test():
    _, target_var = prep.test()
    h = encoder.test()
    decoder2 = AttnDecoderLSTM()
    loss = 0
    Pvocab, outputs = decoder2(target_var,h)
    criterion = nn.NLLLoss()
    for i in range(len(outputs)):
        loss += criterion(Pvocab[i],target_var[i])
    print (loss/len(target_var))


if __name__ == "__main__": 
    test()
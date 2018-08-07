import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from constants import *

class DecoderLSMT3(nn.Module):

	def __init__(self, input_size, output_size, max_length=MAX_LENGTH):
		super(DecoderLSTM3, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size

		self.embedding = nn.Embedding(output_size,hidden_size)
		self.attn = nn.Linear(self.hidden_size *2, self.max_length)


	def forward(self, input, hidden):



import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import argparse

import prepare_dataset as prep
from encoder import EncoderLSTM
from decoder import AttnDecoderLSTM
from constants import *
import save_as_pickle


####################################################################################
##		Prepare Dataset															  ##
####################################################################################

def prepare_dataset():

	dataset_train = prep.loadDataset(path_train)
	dataset_train_tokenized = prep.tokenizeDataset(dataset_train,vocab,buildvocab=True)
	save_as_pickle.save_obj(dataset_train_tokenized,'dataset_train_tokenized') 

	dataset_dev = prep.loadDataset(path_dev)
	dataset_dev_tokenized = prep.tokenizeDataset(dataset_dev)
	save_as_pickle.save_obj(dataset_dev_tokenized,'dataset_dev_tokenized') 

	dataset_eval = prep.loadDataset(path_eval)
	# dataset_eval_tokenized = prep.tokenizeDataset(dataset_eval)
	dataset_eval_tokenized = None
	save_as_pickle.save_obj(dataset_eval_tokenized,'dataset_eval_tokenized') 

	save_as_pickle.save_obj(prep.vocab,'vocab')

####################################################################################
##		Load Dataset															  ##
####################################################################################

def load_dataset():

	dataset_dev = prep.loadDataset(path_dev)
	dataset_train = prep.loadDataset(path_train)
	dataset_eval = prep.loadDataset(path_eval)

	dataset_train_tokenized = save_as_pickle.load_obj('dataset_train_tokenized')
	dataset_dev_tokenized = save_as_pickle.load_obj('dataset_dev_tokenized')
	dataset_eval_tokenized = save_as_pickle.load_obj('dataset_eval_tokenized')

	return dataset_train, dataset_dev, dataset_eval, dataset_train_tokenized, dataset_dev_tokenized, dataset_eval_tokenized

####################################################################################
##		Define training step	 												  ##
####################################################################################

def timeSince(since):
	now = time.time()
	s = now - since
	m = math.floor(s/60)
	s -= m*60
	return '%dm %ds' %(m,s)

def trainingStep(input_var, target_var, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, debug=False):

	if debug:
		print('Starting encoding')

	input_length = len(input_var)
	encoder_hidden = encoder.initHidden()        
	h = Variable(torch.zeros(input_length, encoder.hidden_size*2))        
	for ei in range(input_length):
		encoder_hidden = encoder(input_var[ei],encoder_hidden)
		h[ei] = torch.cat((encoder_hidden[0][0],encoder_hidden[1][0]),1)

	target_length = len(target_var)    
	if debug:
		print('Starting decoding, target_length = %d'%target_length)

	Pvocab, outputs = decoder(target_var,h)
	loss = 0
	for i in range(target_length):
		loss += criterion(Pvocab[i],target_var[i])        

	if debug:
		start = time.time()
		print('Starting backprop')
	loss.backward()    
	if debug:
		print('Done with backprop, took %s'%timeSince(start))

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.data[0] / target_length


def testTrainingStep():
	input_var, target_var = prep.test()
	encoder_opt = optim.SGD(encoder.parameters(), lr=0.01)
	decoder_opt = optim.SGD(encoder.parameters(), lr=0.01)
	encoder = EncoderLSTM()
	decoder = AttnDecoderLSTM()
	loss = trainingStep(input_var,target_var,encoder,decoder,encoder_opt,decoder_opt,nn.NLLLoss(),debug=True)
	print(loss)


####################################################################################
##		Define training process	 												  ##
####################################################################################

plot_losses = []

def showPlot(points):
	plt.figure()
	plt.plot(points)
	plt.show()

def trainIters(dataset_tokenized, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
	start = time.time()
	print_loss_total = 0 # reset every print_every
	plot_loss_total = 0 # reset every plot_every

	encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate)
	training = [random.choice(dataset_tokenized) for i in range(n_iters)]
	criterion = nn.NLLLoss()

	print('\nStarting traning, will print every %d iterations'%print_every)

	for iter in range(1, n_iters + 1):
		# print(iter)
		example = training[iter]
		input_variable = [item for sublist in example['passages'] for item in sublist]
		input_variable = list(input_variable[:400])
		input_variable = prep.variableFromSentence(input_variable,vocab)
		target_variable = example['wellFormedAnswers'][0]
		target_variable = prep.variableFromSentence(target_variable,vocab)

		loss = trainingStep(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
		print_loss_total += loss
		plot_loss_total += loss


		if iter % print_every == 0:
			print(example['wellFormedAnswers'][0])
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print('%s (%d %d%%) %.4f' % (timeSince(start),
		                             iter, iter / n_iters * 100, print_loss_avg))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	showPlot(plot_losses)


def train(dataset):
	encoder1 = EncoderLSTM()
	attn_decoder1 = AttnDecoderLSTM()
	all_losses = trainIters(dataset, encoder1, attn_decoder1, 7500, learning_rate=0.001, print_every=50, plot_every = 1)


if __name__ == "__main__": 
	parser = argparse.ArgumentParser(description='Mode: what do you want to do? Mode can be prep (prepare dataset) or train.')
	parser.add_argument('mode', metavar='mode', type=str, nargs = '?', default='train', help='Mode can be prep (prepare dataset) or train.')
	args = parser.parse_args()

	vocab = prep.createVocabObj(args.mode)

	if args.mode == 'train':
		dataset_train, dataset_dev, dataset_eval, dataset_train_tokenized, dataset_dev_tokenized, dataset_eval_tokenized = load_dataset()
		train(dataset_train_tokenized)
	elif args.mode == 'prep':
		prepare_dataset()
	else:
		print('Not a valide mode!')

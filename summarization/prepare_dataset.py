import json
import random
import re
import math
import sys
import torch
from torch.autograd import Variable

from vocab import Vocab
from mytokenize import *

from constants import *
import save_as_pickle
from helper_functions import *


def createVocabObj(mode='prep'):
	# if mode == 'prep':
	vocab = Vocab() 
	return vocab
	# else:
	# 	vocab = save_as_pickle.load_obj('vocab')
	# return vocab

def loadDataset(path,limit=100000000000):
	print("Loading dataset from %s..."%path)
	dataset = list()
	i = 0
	for line in open(path, 'r'):
		all_data = json.loads(line)
		for data in all_data:
			if i <limit:
				dataset.append(data)
				i += 1
			else:
				break
	print("Loaded %d pieces of data"%len(dataset))
	return dataset


def randomExamples(dataset):
	for i in range(6):
		print('')
		x = random.choice(dataset)
		print(x['query'])
		print(x['wellFormedAnswers'][0])


def tokenizeDataset(dataset,vocab,buildvocab=False):

	tokenized_dataset = list()
	len_dataset = len(dataset)
	print_every = math.floor(len_dataset / 10)
	print('Tokenizing dataset...')
	for i, data in enumerate(dataset):
		tokenized_data = dict()
		tokenized_data['answers'] = list()
		tokenized_data['wellFormedAnswers'] = list()
		tokenized_data['passages'] = list()
		for answer in data['answers']:
			t = tokenizeSentence(answer)
			tokenized_data['answers'].append(t)
			if buildvocab:
				vocab.addSentence(t)
		for wf_answer in data['wellFormedAnswers']:
			t = tokenizeSentence(wf_answer)
			tokenized_data['wellFormedAnswers'].append(t)
			if buildvocab:
				vocab.addSentence(t)
		for passage in data['passages']:
			t = tokenizeSentence(passage['passage_text'])
			tokenized_data['passages'].append(t)
			if buildvocab:
				vocab.addSentence(t)
		if i>0 and i%print_every == 0:
			sys.stdout.write('... %d%%\r'%(i/print_every*10))
			sys.stdout.flush()
		tokenized_dataset.append(tokenized_data)

	print('Number of different words: %d'%len(vocab.word2count))

	if buildvocab:
		vocab.buildVocabulary()

	return tokenized_dataset

def indexesFromSentence(sentence,vocab): # sentence is a list of tokens
	return [[vocab.getIndex(word)] for word in sentence]

# def variableFromSentence(sentence,vocab):
# 	indexes = indexesFromSentence(sentence,vocab)
# 	indexes.append(EOS_token)
# 	return Variable(torch.LongTensor(indexes).view(-1,1))

def tensorFromSentence(sentence,vocab):
	indexes = indexesFromSentence(sentence,vocab)
	# print(indexes)
	indexes.append([EOS_token])
	# return torch.LongTensor(indexes, device=DEVICE).cuda()
	return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(-1,1)

# def variablesFromPair(pair,vocab):
# 	input_var = variableFromSentence(pair[0],vocab)
# 	target_var = variableFromSentence(pair[1],vocab)
# 	return input_var, target_var

# def tensorsFromPair(pair,vocab):
# 	input_var = tensorFromSentence(pair[0],vocab)
# 	target_var = tensorFromSentence(pair[1],vocab)
# 	return input_var, target_var

def testTokenizeVsLoadingTime(dataset,name,vocab):
	s1 = time.time()
	dataset_tokenized = tokenizeDataset(dataset,vocab)
	print('It took %s to tokenize the dataset'% timeSince(s1))
	s2 = time.time()
	dataset_tokenized2 = save_as_pickle.load_obj(name)
	print('It took %s to load the tokenized dataset'% timeSince(s2))

def test(path,verbose=False):
	# dataset_train_tokenized = tokenizeDataset(dataset_train)
	vocab = createVocabObj()
	dataset_dev = loadDataset(path)
	dataset_dev_tokenized = tokenizeDataset(dataset_dev,vocab,buildvocab=True)

	example = random.choice(dataset_dev_tokenized)
	print(example)
	# print(example['passages'])
	input_var = [item for sublist in example['passages'] for item in sublist]
	input_var = list(input_var[:400])
	input_var = tensorFromSentence(input_var,vocab)
	# input_var = input_var.to(DEVICE)
	target_var = example['wellFormedAnswers'][0]
	target_var = tensorFromSentence(target_var,vocab)
	# target_var = target_var.to(DEVICE)
	if verbose:
		print('Size input var')
		print(input_var.size())
		print('Size target var')
		print(target_var.size())
	return input_var, target_var

if __name__ == "__main__": 
	path = path_dev
	# testTokenizeVsLoadingTime(loadDataset(path),'dataset_train_tokenized',createVocabObj())
	test(path,True)
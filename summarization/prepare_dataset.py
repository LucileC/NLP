import json
import random
import re
import math
import torch
from torch.autograd import Variable

from vocab import Vocab
from mytokenize import *


def loadDataset(path,limit=100000000000):
	print("Loading dataset...")
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


def tokenizeDataset(dataset):

	vocab = Vocab()

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
			vocab.addSentence(t)
		for wf_answer in data['wellFormedAnswers']:
			t = tokenizeSentence(wf_answer)
			tokenized_data['wellFormedAnswers'].append(t)
			vocab.addSentence(t)
		for passage in data['passages']:
			t = tokenizeSentence(passage['passage_text'])
			tokenized_data['passages'].append(t)
			vocab.addSentence(t)
		if i>0 and i%print_every == 0:
			print('... %d%%\r'%(i/print_every*10))
		tokenized_dataset.append(tokenized_data)

	print('Number of different words: %d'%len(vocab.word2count))

	vocab.buildVocabulary()

	return tokenized_dataset

def indexesFromSentence(sentence): # sentence is a list of tokens
	return [vocab.word2index[word] for word in sentence]

def variableFromSentence(sentence):
	indexes = indexesFromSentence(sentence)
	indexes.append(EOS_token)
	return Variable(torch.LongTensor(indexes).view(-1,1))

def variablesFromPair(pair):
	input_var = variableFromSentence(pair[0])
	target_var = variableFromSentence(pair[1])
	return input_var, target_var

def test():
	path_dev = "data/msmarco_2wellformed/dev_v2.0_well_formed.json"
	dataset_train_tokenized = tokenizeDataset(dataset_train)
	dataset_dev = loadDataset(path_dev)
	dataset_train_tokenized = tokenizeDataset(dataset_train)
	example = random.choice(dataset_train_tokenized)
	# print(example['passages'])
	input_var = [item for sublist in example['passages'] for item in sublist]
	input_var = list(input_var[:400])
	input_var = variableFromSentence(input_var)
	print(input_var.size())
	target_var = example['wellFormedAnswers'][0]
	target_var = variableFromSentence(target_var)
	print(target_var.size())
	return input_var, target_var

if __name__ == "__main__": 
	test()
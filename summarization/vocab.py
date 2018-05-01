from constants import *

class Vocab:
    def __init__(self,voc_size=VOC_SIZE):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK"}
        self.wordsOrderedByFreq = list()
        self.word2count["UNK"] = 0
        self.n_words = 3
        self.voc_size = voc_size
        
    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)
            
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
        else:
            self.word2count[word] += 1
            
    def buildVocabulary(self):
        # sort words by frequencies
        self.wordsOrderedByFreq = list(reversed(sorted(self.word2count, key=lambda key: self.word2count[key])))
        # get voc_size th element in list (= last word in vocab)
        last_word = self.wordsOrderedByFreq[self.voc_size]
        # get frequency of last word
        freq = self.word2count[last_word]
        print("Last word in vocabulary will be %s with a frequency of appearance of %d"%(last_word,freq))
        for i in range(self.voc_size):
            word = self.wordsOrderedByFreq[i]
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        for j in range(self.voc_size,len(self.wordsOrderedByFreq)):
            self.word2index[word] = 2
            self.n_words += 1
            self.word2count['UNK'] += 1
        print("Vocabulary (size %d) is built. The firsts words are:"%self.voc_size)
        for k in range(2,7):
            word = self.index2word[k]
            print("%s (appeared %d times)"%(word,self.word2count[word]))
import re

def insertCharIfSeq(sentence1,c,seq):
    i = 0
    indexes = [m.start() for m in re.finditer(seq, sentence1)]
    for index in indexes:
        sentence1 = sentence1[:index+i] + c + sentence1[index+i:]
        i += 1
    return sentence1

def processContractions(sentence1):
    sentence1 = insertCharIfSeq(sentence1," ","'s")
    sentence1 = insertCharIfSeq(sentence1," ","'m")
    sentence1 = insertCharIfSeq(sentence1," ","'ll")
    sentence1 = insertCharIfSeq(sentence1," ","'ve")
    sentence1 = insertCharIfSeq(sentence1," ","'re")
    sentence1 = insertCharIfSeq(sentence1," ","'d")
    return sentence1


def processNegatives(sentence1):
    i = 0
    indexes = [m.start() for m in re.finditer("can't", sentence1)]
    for index in indexes:
        sentence1 = sentence1[:index+i+3] + sentence1[index+i+2:]
        i += 1
    return insertCharIfSeq(sentence1," ","n't")

## WHAT TO DO WITH HYPHENS ??

def tokenizeSentence(sentence1):
#     processFinalPeriod(sentence1)
    sentence1 = processContractions(sentence1)
    sentence1 = processNegatives(sentence1)
    s = sentence1.lower()
    s = re.sub('''([.,!"?$;:/#`()])''', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    s = s.split()
#     s = processHyphenIfUnknownWords(s,glove)
#     s.append('</s>')
#     s = ['<s>'] + s
    return s

def testTokenizeSentence():
    sentence1 = "I can't open the door because the 30-year old/blond-hair guy doesn't want to let me in. He's mean, isn't he? I can't go in! There's no other way!"
    words = tokenizeSentence(sentence1)
    print(words)


if __name__ == "__main__": 
    testTokenizeSentence()
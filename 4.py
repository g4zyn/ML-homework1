import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from nltk import FreqDist
from nltk.tokenize import wordpunct_tokenize
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from unidecode  import unidecode  

import math

import os

BOW_FEATURE_SIZE =10000
FREQUENT_K = 5
class_name = ["Negative","Positive"]


class MultinomialNaiveBayes:
    def __init__(self,nb_classes,nb_words,pseudocount):
        self.nb_classes = nb_classes
        self.nb_words = nb_words
        self.pseudocount = pseudocount
        


    def fit(self,X,Y):
        nb_examples = X.shape[0]
        self.prioris = np.bincount(Y)/nb_examples
        
        occs = np.zeros((self.nb_classes,self.nb_words))
        for i in range(nb_examples):
            c = Y[i]
            for w in range(self.nb_words):
                cnt = X[i][w]
                occs[c][w]+=cnt
        self.like = np.zeros((self.nb_classes,self.nb_words))
        for c in range(self.nb_classes):
            for w in range(self.nb_words):
                up = occs[c][w] + self.pseudocount
                down = np.sum(occs[c])+self.nb_words*self.pseudocount
                self.like[c][w] = up/down
        

    def predict(self,BOW):
        probs = np.zeros(self.nb_classes)
        for c in range(self.nb_classes):
            prob = np.log(self.prioris[c])
            for w in range(self.nb_words):
                cnt = BOW[w]
                prob+=cnt*np.log(self.like[c][w])
            probs[c] = prob

        prediction = np.argmax(probs)
        return prediction

    def predict_multiply(self,BOW):
        probs = np.zeros(self.nb_classes)

        for c in range(self.nb_classes):
            prob = self.prioris[c]
            for w in range(self.nb_words):
                cnt = BOW[w]
                prob*= self.like[c][w]**cnt

            probs[c] = prob

        prediction = np.argmax(probs)
        return prediction

def occ_score(word,doc):
    return 1 if word in doc else 0

def numocc_score(word,doc):
    return doc.count(word)

def freq_score(word,doc):
    return doc.count(word)/len(doc)
    
def create_vocabulary(corpus):
    print("Creatin vocab...")
    vocab_d = {}
    for d in corpus:
        for word in d:
            vocab_d.setdefault(word,0)
            vocab_d[word]+=1
    print("Vocab finished")
    return sorted(vocab_d,key=vocab_d.get,reverse=True)[:BOW_FEATURE_SIZE]

def create_bow(doc,vocabulary):
    bow = np.zeros(len(vocabulary),dtype=np.float64)
    for word_index in range(len(vocabulary)):
        word = vocabulary[word_index]
        cnt = numocc_score(word,doc)
        bow[word_index] = cnt
        

    return bow

def BoW_model(corpus,labels,vocabulary):
    print("Creating bow features...")
    X = np.zeros((len(corpus),len(vocabulary)),dtype=np.float64)
    i=0
    for dci in range(len(corpus)):
        doc = corpus[dci]
        X[dci] = create_bow(doc,vocabulary)
        i+=1
        print(i)
    Y = np.zeros(len(corpus),dtype=np.int32)
    for j in range(len(Y)):
        Y[j] = labels[j]
    print("finished bow")
    return X,Y

def remowe_non_ascii(text):
    return ''.join([x for x in text if ord(x) < 128])

def get_data(file_name):
    positives = []
    nregatives = []
    all_corpus = []
    labels = []
    
   # positive_corpus= []
   # negative_corpus = []

    with open(file_name,encoding='utf8') as fp:
        next(fp)
        for line in fp:
            a = line.split(',')
            if a[1] == '1':
                labels.append(1)
            elif a[1] == '0':
                labels.append(0)
            all_corpus.append(remowe_non_ascii(','.join(a[2:])))
            

    # mesanje podataka
    
    indexes = np.random.permutation(len(all_corpus))
    all_corpus = np.asarray(all_corpus)
    all_corpus = all_corpus[indexes]

    labels = np.asarray(labels)
    labels = labels[indexes]


    return all_corpus,labels

def clean_data(data):
    clean_corpus = []
    porter = PorterStemmer()
    print("corpus cleaning")
    stop_punc = set(stopwords.words('english')).union(set(punctuation))
    for c in data:
        words = wordpunct_tokenize(c)
        table = str.maketrans('', '', punctuation)
        words_lower = [w.lower() for w in words]
        words_stripped = [w.translate(table) for w in words_lower]  
        words_filtered = [w for w in words_stripped if w not in stop_punc and w.isalpha()]
        words_stemmed = [porter.stem(w) for w in words_filtered]
        clean_corpus.append(words_stemmed)

    return clean_corpus

def five_most_freq(clean_corpus,labels,class_type):
    dic = {}
    counter = 0

    for i in range(len(clean_corpus)):
        if labels[i] == class_type:
            counter+=1
            for w in clean_corpus[i]:
                dic.setdefault(w,0)
                dic[w] +=1

    return sorted(dic,key=dic.get,reverse=True)[:FREQUENT_K]

def LR(word,labels,clean_corpus):

    porter = PorterStemmer()
    word = porter.stem(word.lower())
    p_count = 0
    n_count = 0

    for i in range(len(clean_corpus)):
        doc = clean_corpus[i]
        if labels[i] == 1:
            p_count+=doc.count(word)
        elif labels[i] == 0:
            n_count+=doc.count(word)

    result = 0

    if p_count > 9 and n_count > 9:
        result =p_count/n_count

    return result,word


def remove_dups(lista):
    return list(dict.fromkeys(lista))

def main():
    file_name = "data/twitter.csv"


    all_corpus,labels = get_data(file_name)
    all_corpus_cleaned = clean_data(all_corpus)

    limit = math.floor(len(all_corpus_cleaned) * 0.8)
    train_corpus = all_corpus_cleaned[:limit]
    train_labels = labels[:limit]

    test_corpus = all_corpus_cleaned[limit:]
    test_labels = labels[limit:]

    vocabulary = create_vocabulary(train_corpus)

    X,Y = BoW_model(train_corpus,train_labels,vocabulary)
    model = MultinomialNaiveBayes(nb_classes=2,nb_words=BOW_FEATURE_SIZE,pseudocount =1)
    model.fit(X,Y)
    correct_pred = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(test_corpus)):

        doc = test_corpus[i]
        label = test_labels[i]
        bow = create_bow(doc,vocabulary)
        prediction = model.predict_multiply(bow)

        if prediction == test_labels[i]:
            correct_pred+=1
        
        if prediction == 1 and test_labels[i] ==1:
            TP+=1
        elif prediction == 1 and test_labels[i] == 0:
            FP+=1
        elif prediction == 0 and test_labels[i] == 0:
            TN+=1
        elif prediction == 0 and test_labels[i] == 1:
            FN+=1


    confusion_matrix = [[TN,FP],[FN,TP]]
    acc = correct_pred/len(test_corpus)

    

    print(acc)
    print(confusion_matrix)


    print("Top five negative: ",five_most_freq(all_corpus_cleaned,labels,0))
    print("Top five positives: ",five_most_freq(all_corpus_cleaned,labels,1))

    lr_predictions = []

    for i in range(len(train_corpus)):
        for word in train_corpus[i]:
            lr_predictions.append(LR(word,labels,train_corpus))


    lr_predictions.sort(key=lambda tup:tup[0])
    lr_predictions = remove_dups(lr_predictions)
    # length = len(lr_predictions)

    # for p in lr_predictions:
    #     print([])


if __name__ == '__main__':
    main()


"""
LR : 7.4 - thank
LR : 5.1 - cool
LR : 4.8 - birthday
LR : 4.6 - cute
LR : 4.3 - happi
LR : 4.2 - follow

LR : 0 - franki
LR : 0 - bandaidedto
LR : 0 - lifeh
LR : 0 - chrisdaughtri
LR : 0 - awwwwww

Najcescih 5 reci u pozitivnim i negativnim kritikama su slicne, par reci su cak iste.
LR metrika je odnos pojave poz i neg reci u kritikama. Visoka LR vrednost znaci da je vezana za pozitivne, dok manja za negativne



"""
#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-03-08 18:19:28
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-15 23:28:47




"""
LSTM based POS tagger
"""

import os
import io
import sys
import pdb
import nltk
import zipfile
import argparse
import numpy as np
import pandas as pd
import time
import urllib.request
from nltk import corpus
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
import seaborn; seaborn.set() # plot formatting
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, Bidirectional
from keras.models import Model
from sklearn.metrics import classification_report

MAX_SEQ_LEN = 100
EMBEDDING_SIZE  = 300

class Vectorizer(object):
    def __init__(self, sentences, pos_tags, word2vec):
        self.sentences = sentences
        self.pos_tags = pos_tags
        self.word2vec = word2vec
    

    def vectorize(self):
        # Vectorize X and Y
        self.t_words = Tokenizer()
        self.t_words.fit_on_texts(self.sentences)  
        X = self.t_words.texts_to_sequences(self.sentences)

        self.t_tags = Tokenizer()
        self.t_tags.fit_on_texts(self.pos_tags)  
        y = self.t_tags.texts_to_sequences(self.pos_tags)
        
        # Padding and truncation to make all sequences of uniform length
        self.X = pad_sequences(X, maxlen=MAX_SEQ_LEN, truncating='post')
        self.y = pad_sequences(y, maxlen=MAX_SEQ_LEN, truncating='post')
        self.embedding_weights = self.prep_embeddings()
        
        self.target_labs = [*self.t_tags.word_index]
        return self.X, self.y, self.target_labs, self.embedding_weights

    def prep_embeddings(self):
    
        Vectorizer.prep_embeddings.VOCABULARY_SIZE = len(self.t_words.word_index) + 1
        self.embedding_weights = np.zeros((Vectorizer.prep_embeddings.VOCABULARY_SIZE, EMBEDDING_SIZE))
        word2id = self.t_words.word_index
        for word, index in word2id.items():
            try:
                self.embedding_weights[index, :] = self.word2vec[word]
            except KeyError:
                pass
        self._encodings()
        return self.embedding_weights

    def _encodings(self):
        # one-hot encoding
        self.y = to_categorical(self.y)
        

class LSTM_model(object):
    def __init__(self, X, y, target_labs, embedding_weights, BiLSTM=False):
        self.X = X
        self.y = y
        self.target_labs = target_labs
        self.embedding_weights = embedding_weights
        self.BiLSTM = BiLSTM
        
    
    def train_test_split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, 
                                                                                self.y, 
                                                                                test_size=0.2, 
                                                                                random_state=123)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, 
                                                                                            self.Y_train,
                                                                                            test_size=0.2, 
                                                                                            random_state=123)
    def model_init(self, args):
        self.lstm_model = Sequential()
        self.target_labs = self.y.shape[2]
        VOCABULARY_SIZE = Vectorizer.prep_embeddings.VOCABULARY_SIZE
        self.lstm_model.add(Embedding(
                                input_dim = VOCABULARY_SIZE,         
                                output_dim  = EMBEDDING_SIZE,        
                                input_length = MAX_SEQ_LEN,          
                                weights = [self.embedding_weights],  
                                trainable = False                     
        ))
        if self.BiLSTM:
            print('-'*20,"Model architecture: Bi-LSTM",'-'*20)
            self.lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
        else:
            print('-'*20,"Model architecture: LSTM",'-'*20)
            self.lstm_model.add(LSTM(64, return_sequences=True))
            
        self.lstm_model.add(TimeDistributed(Dense(self.target_labs, activation='softmax')))
        path = args.out_dir + "/model_arch.png"
        keras.utils.plot_model(self.lstm_model, path, show_shapes=True)
        

    def train(self, args):
        self.train_test_split()
        self.model_init(args)
        self.lstm_model.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['acc'])
        print(self.lstm_model.summary())
        self.lstm_training = self.lstm_model.fit(self.X_train,
                                                self.Y_train, 
                                                batch_size=128, 
                                                epochs=10, 
                                                validation_data=(self.X_validation, self.Y_validation))
        self.training_report(args)
        loss, accuracy = self.lstm_model.evaluate(self.X_test, self.Y_test, verbose = 1)
        print("Loss: {:.2f} %,\nAccuracy: {:.2f} %".format(loss*100, accuracy*100))
        
        if self.BiLSTM:
            self.lstm_model.save(args.out_dir + '/model_Bi-LSTM.h5')
        else:
            self.lstm_model.save(args.out_dir + '/model_LSTM.h5')
            
    def training_report(self, args):
                
        # summarize history for accuracy
        plt.plot(self.lstm_training.history['acc'])
        plt.plot(self.lstm_training.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.BiLSTM:
            plt.savefig(args.out_dir + '/accuray_plot_Bi-LSTM.eps', format='eps')
        else:
            plt.savefig(args.out_dir + '/accuray_plot_LSTM.eps', format='eps')
        
        # summarize history for loss
        plt.plot(self.lstm_training.history['acc'])
        plt.plot(self.lstm_training.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if self.BiLSTM:
            plt.savefig(args.out_dir + '/loss_plot_Bi-LSTM.eps', format='eps')
        else:
            plt.savefig(args.out_dir + '/loss_plot_LSTM.eps', format='eps')
    
    
def download_data(args):
    """ purpose of my function """

    # Download dataset
    nltk.download('treebank')
    nltk.download('brown')
    nltk.download('conll2000')
    nltk.download('universal_tagset')

    # Download pre-trained embeddings. 1M words with subword info
    if args.subword_info:
        print("Using Subword info")
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip'
    else:
        print("Not using Subword info")
        url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
    fname = url.split('/')[-1]
    if not os.path.exists(fname):
        print("Downloading embeddings...")
        command = str("! wget -nc ") + url
        print(command)
        os.system(command)
        with zipfile.ZipFile(fname, 'r') as zip_ref:
            zip_ref.extractall('.')
    else:
        print("Pre-trained embeddings already present; not retrieving.")
    
    return KeyedVectors.load_word2vec_format(fname.strip('.zip'))


def data_preparation():
    treebank_corpus = corpus.treebank.tagged_sents(tagset='universal')
    brown_corpus = corpus.brown.tagged_sents(tagset='universal')
    conll_corpus = corpus.conll2000.tagged_sents(tagset='universal')
    return treebank_corpus + brown_corpus + conll_corpus
    

def data_loader(tagged_sentences):
    sentences = []
    pos_tags = []
    for item in tagged_sentences:
        sentences.append([token[0] for token in item])
        pos_tags.append([token[1] for token in item])
    return sentences, pos_tags

    
def main():
    """ main method """
    args = parse_arguments()
    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    word2vec = download_data(args)
    tagged_sentences = data_preparation()
    sentences, pos_tags = data_loader(tagged_sentences)
    vectorizer = Vectorizer(sentences, pos_tags, word2vec)
    X, y, target_labs, embedding_weights = vectorizer.vectorize()
    if args.model_choice=="BiLSTM":
        LSTM_model(X, y, target_labs, embedding_weights,BiLSTM=True).train(args)
    elif args.model_choice=="LSTM":
        LSTM_model(X, y, target_labs, embedding_weights,BiLSTM=False).train(args)
    else:
        print("Specified model architecture not understood. Try again!")
    end = time.time()
    print("Total runtime: %.3f s" % (end-start))
    

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", help="path to save the plots and results")
    parser.add_argument("model_choice",default=None,choices=['LSTM','BiLSTM'], type=str,help='Choice of training architecture')
    parser.add_argument("-subword_info", default=None, type=bool, help='use pre-embeddings with subword information')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()


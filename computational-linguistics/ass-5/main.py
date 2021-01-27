#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-25 18:39:24
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-27 22:33:23



"""
<Function of script>
"""

import os
import sys
import time
import pdb
import argparse
import numpy as np
from collections import defaultdict
import logging
from itertools import chain


class LDA_Gibbs(object):
    def __init__(self, alpha, beta, num_topics, 
                epochs, num_top_words, vocab, vocab_id_dict, docs):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.epochs = epochs
        self.vocab = vocab
        self.vocab_id_dict = vocab_id_dict
        self.docs = docs
        self.num_top_words = num_top_words
    
    def random_initialization(self):
        """go through each document and randomly assign 
        each word in the document to one of the topics"""
        start = time.time()
        self.num_words = len(self.vocab)
        self.num_docs = len(self.docs)
        
        # word-topic distribution matrix: Topics X Words
        self.wt = np.zeros(shape=(self.num_topics, self.num_words))
        
        # word-topic assignment:  Docs X Words
        self.t = np.zeros(self.num_topics)
        self.d = np.zeros(self.num_docs)
        
        # document-topic distribution: Docs X Topics
        self.dt = np.zeros(shape=(self.num_docs, self.num_topics))
        self.topics = defaultdict(int)
        # randomly assign a topic to each word w in a document
        for doc_id in range(self.num_docs):
            for word_id, word in enumerate(self.docs[doc_id]):
                z = np.random.randint(self.num_topics)
                self.dt[doc_id, z] += 1
                self.d[doc_id] += 1
                self.wt[z, word] += 1
                self.t[z] += 1
                self.topics[(doc_id, word_id)] = z
        end = time.time()
        logging.info("Initialization runtime: {:.3f} s\n".format(end - start))
        
    def gibbs_sampler(self, w, d):
        
        prob_z = (self.wt[:, w] + self.beta) / (self.t + (self.num_words * self.beta)) * \
            (self.dt[d, :] + self.alpha) / (self.d[d] + (self.num_topics * self.alpha))
        # Normalization
        prob_z /= np.sum(prob_z)
        return prob_z

    def train(self):

        self.random_initialization()
        
        for epoch in range(self.epochs):
            start = time.time()
            for doc_id in range(self.num_docs):
                for word_id, word in enumerate(self.docs[doc_id]):
                    z = self.topics[(doc_id, word_id)]
                    self.dt[doc_id, z] -= 1
                    self.d[doc_id] -= 1
                    self.wt[z, word] -= 1
                    self.t[z] -= 1
                    prob_z = self.gibbs_sampler(word, doc_id)
                    # sampling
                    z = np.random.choice(np.arange(self.num_topics), p=prob_z)
                    self.topics[(doc_id, word_id)] = z
                    self.dt[doc_id, z] += 1
                    self.d[doc_id] += 1
                    self.wt[z, word] += 1
                    self.t[z] += 1
            end = time.time()
            logging.info("Epoch: {}/{}, runtime: {:0.3f} s".format(epoch+1, self.epochs, end-start))
        
        word_list = self.get_topicwise_words()
        return word_list
    
    def get_topicwise_words(self):
        self.wt_sorted = (-self.wt).argsort()[:, :self.num_top_words]
        self.vocab_id_dict = {v: k for k, v in self.vocab_id_dict.items()}
        word_list = [ self.idx2word(self.wt_sorted[topic_id]) for topic_id in range(self.num_topics)]

        return word_list
        
    def idx2word(self, idx_list):
        return [self.vocab_id_dict[x] for x in idx_list]
    

def data_loader(args):
    """ purpose of my function """
    
    start= time.time()
    doc_list = [line.split() for line in open(args.corpus_f)][1:][:args.num_sents]
    # list of all vocabs in corpus
    vocab = list(chain.from_iterable(doc_list))
    vocab_id_dict = {}
    for idx, word in enumerate(vocab):
        if word not in vocab_id_dict:
            vocab_id_dict[word] = idx
    docs = []
    for doc in doc_list:
        temp = [ vocab_id_dict[word] for word in doc]
        docs.append(np.asarray(temp))
    end = time.time()
    logging.info("\nData loading runtime: %.3f s" % (end - start))
    return vocab, vocab_id_dict, docs
    
    

def main():
    """ main method """
    args = parse_arguments()
    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    
    logging.basicConfig(format='%(asctime)s %(message)s',
                    filename=args.out_dir + '/' + 'training.log', filemode='a',
                    level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Expected runtime: 3.3 hrs")
    
    vocab, vocab_id_dict, docs = data_loader(args)
    lda_gibbs = LDA_Gibbs(args.alpha, args.beta, args.num_topics, 
                args.epochs, args.num_top_words, vocab, vocab_id_dict, docs)
    word_list = lda_gibbs.train()
    outfile = str(args.out_dir) + "/topicwise_words.txt"
    myfile = open(outfile, 'w')
    for idx, words in enumerate(word_list):
        words = ', '.join(map(str, words))
        myfile.write("Topic %i: %s. " % (idx+1, words))
        myfile.write("\n")
    myfile.close()
    end = time.time()
    tot_rt = (end - start)/3600
    logging.info('%s %.3f %s', 'Total runtime:',tot_rt, 'hrs.')
    logging.info('%s: %s%s','Results saved to:', args.out_dir, '/topicwise_words.txt')
    logging.info('%s', '-- Done --')
    

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_f", help="path to input text corpus file")
    parser.add_argument("out_dir", help="path to save the frequent words for each topic")
    parser.add_argument("-alpha", default=0.02, type=float, help='Parameter that sets the topic distribution for the documents')
    parser.add_argument("-beta", default=0.1, type=float, help='Parameter that sets the topic distribution for the words')
    parser.add_argument("-num_topics", default=20, type=int, help='number of topics')
    parser.add_argument("-epochs", default=500, type=int, help='number of training iterations')
    parser.add_argument("-num_sents", default=2001, type=int, help='number of sentences to train from')
    parser.add_argument("-num_top_words", default=10, type=int, help='number of words to select from each topic')
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()


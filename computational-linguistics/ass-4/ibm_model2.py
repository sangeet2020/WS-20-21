#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-10 16:35:59
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-12 14:12:55




"""
<Function of script>
"""

import os
import sys
import argparse
import math
import numpy as np
import operator
import nltk
import time
from sys import getsizeof
from collections import defaultdict
from itertools import product


class IBM_model(object):
    
    def __init__(self, english_words, foreign_words, sentence_pairs):
        self.english_words = english_words
        self.foreign_words = foreign_words
        self.sentence_pairs = sentence_pairs
    
    def initial_probablity(self):
        print("Obtaining inital probablity...")

        t_score = np.full((len(self.english_words), len(self.foreign_words)), 1 / (len(self.foreign_words)))
        print(sys.getsizeof(t_score)/1e+9,"GB")
        
        self.t_score = t_score
        # for i in self.t_score:
        #         print(i)
        # print("*"*40)
        
    
    def _return_index(self, e_word, f_word):
        # Return the index in the form of tuple (e_idx, f_idx)
        # so as to append the score in the t_score matrix
        e_idx, f_idx = (self.english_words.index(e_word), self.foreign_words.index(f_word))
        return e_idx, f_idx
    
    def cond_prob_e_f(self, e_words, f_words): # P(e|f)
        le = len(e_words)
        lf = len(f_words)
        
        epsilon = 1 # Normalization constant
        
        N_factor = epsilon / ((lf)**le)
        product = 1
        
        for j in range(le):
            inner_sum = 0
            for i in range(le):
                try:
                    e_idx, f_idx = self._return_index(e_words[j], f_words[i])
                except IndexError:
                    import pdb; pdb.set_trace()
                inner_sum += self.t_score[(e_idx, f_idx)]
            
            product = inner_sum * product
        
        p_e_f = N_factor*product
        return p_e_f
    
    
    def perplexity(self):
        pp = 0
        for pair in self.sentence_pairs:
            f_words = pair[0]
            e_words = pair[1]
            # print('english sentence:', e_words, 'foreign sentence:', f_words)
            
            pp += -math.log(self.cond_prob_e_f(e_words, f_words), 2)
        
        pp = 2**pp
        return pp

    def aligner(self, epochs):
        print("EM algorithm...")
        self.initial_probablity()
        for _ in range(epochs):
            # pp = self.perplexity()
            # print("Perplexity: {}".format(pp))
            
            # Initialize
            # total = {}
            count = np.zeros((len(self.english_words), len(self.foreign_words)))
            total = np.zeros(len(self.foreign_words))
            # for f_word in self.foreign_words:
            #     total[f_word] = 0
            #     for e_word in self.english_words:
            #         count[(e_word, f_word)] = 0
            
            print("E- step...")
            for sp in self.sentence_pairs:
                f_words = sp[0]
                e_words = sp[1]
                
                # Compute normalizations
                # s_total = dict()
                s_total = np.zeros(len(self.english_words))
                for e, f in product(e_words, f_words):
                    # s_total[e] = 0
                    e_idx, f_idx = self._return_index(e, f)
                    s_total[self.english_words.index(e)] += self.t_score[(e_idx, f_idx)]
                
                
                # Collect counts
                for e, f in product(e_words, f_words):
                    e_idx, f_idx = self._return_index(e, f)
                    count[(e_idx, f_idx)] += self.t_score[(e_idx, f_idx)] / s_total[self.english_words.index(e)]
                    total[self.foreign_words.index(f)] += self.t_score[(e_idx, f_idx)] / s_total[self.english_words.index(e)]
            
            print("M- step...")
            # Estimate probablities
            for f_word, e_word in product(self.foreign_words, self.english_words):
                # for e_word in self.english_words:
                e_idx, f_idx = self._return_index(e_word, f_word)
                self.t_score[(e_idx, f_idx)] = count[(e_idx, f_idx)] / total[self.foreign_words.index(f_word)]
        
        # print(self.t_score)
            
    def extractor(self):
        # Extracting alignments from probablity scores
        threshold=0.01
        alignments = {}
        for e_word in self.english_words:
            sig_score = {}
            for word_pair, score in self.t_score.items():
                if word_pair[0] == e_word:
                    sig_score[word_pair[1]] = score
            # import pdb; pdb.set_trace()
            threshold_query = dict((k, v) for k, v in sig_score.items() if v >= threshold)
            alignments[e_word] = [*threshold_query.keys()]
            # import pdb; pdb.set_trace()
        # print(alignments)
        # print("Prob dict: ", self.t_score)
        return alignments
    
    
                    

def my_function(arg_1, arg_2, args):
    """ purpose of my function """

def tokenize(text):
    f_tokens = nltk.word_tokenize(text[0].lower())
    e_tokens = nltk.word_tokenize(text[1].lower())
    return f_tokens, e_tokens


def main():
    """ main method """
    args = parse_arguments()
    # os.makedirs(args.out_dir, exist_ok=True)
    # sentence_pairs = [ 
    #     [ ['das', 'Haus'], ['the', 'house'] ], 
    #     [ ['das', 'Buch'], ['the', 'book'] ], 
    #     [ ['ein', 'Buch'], ['a', 'book'] ]
    # ]
    
    e_file = "jhu-mt-hw/hw2/data/hansards.e"
    f_file = "jhu-mt-hw/hw2/data/hansards.f"
    e_file = "hansards.e"
    f_file = "hansards.f"
    sentence_pairs = []
    start = time.time()
    with open(f_file, 'r') as f_fpw, open(e_file, 'r') as e_fpw:
        file_iter = list(zip(f_fpw, e_fpw))
        for line in file_iter:
            f_tokens, e_tokens = tokenize(line)
            sentence_pairs.append([f_tokens, e_tokens])    
    end=  time.time()
    print("Total runtime data loading: %.3f s" % (end-start))
    
    # sentence_pairs = [ 
    #     [ ['la', 'maison'], ['the', 'house'] ], 
    #     [ ['la', 'maison', 'bleu'], ['the', 'blue', 'house'] ], 
    #     [ ['la', 'fleur'], ['the', 'flower'] ]
    # ]
    
    # import pdb; pdb.set_trace()
    print("No. of sentence of pairs= ", len(sentence_pairs))
    print("*"*40)
    # print("Sentence pairs:")
    # for pair in sentence_pairs:
    #     print(pair)
    english_words = []
    foreign_words = []
    for pair in sentence_pairs:
        foregin_pair = pair[0]
        english_pair = pair[1]
        for e_word in english_pair:
            english_words.append(e_word)
        for f_word in foregin_pair:
            foreign_words.append(f_word)
    
    english_words = sorted(list(set(english_words)), key=lambda s: s.lower()) 
    foreign_words = sorted(list(set(foreign_words)), key=lambda s: s.lower()) 
    # print("*"*40)
    print("Foreign words: ", len(foreign_words))
    # print("*"*40)
    print("English words: ", len(english_words))
    # print("*"*40)
    ibm_model = IBM_model(english_words, foreign_words, sentence_pairs)
    
    
    
    start = time.time()
    ibm_model.aligner(epochs=1)
    end = time.time()
    print("Total runtime aligner: %.3f s" % (end-start))
    import pdb; pdb.set_trace()
    start = time.time()
    alignments = ibm_model.extractor()
    end = time.time()
    print("Total runtime extractor: %.3f s" % (end-start))
    # print(alignments)
    # for i,v in alignments.items():
    #     print(i,v)
    
    
    """ Mappings:
    {'blue': 'bleu', 'flower': 'fleur', 'house': 'maison', 'the': 'la'}
    """
    print("*"*40)
    # print("Alignments: ", alignments)
    print("*"*40)
    for sp in sentence_pairs:
        foreign_sent = sp[0]
        english_sent = sp[1]
        for e_word in english_sent:
            e_idx = english_sent.index(e_word)+1
            get_f_words = alignments.get(e_word)
            try:
                # import pdb; pdb.set_trace()
                for word in get_f_words:
                    f_idx = foreign_sent.index(word)+1
                    sys.stdout.write("%i-%i " % (e_idx,f_idx))
            except ValueError:
                f_idx = 0
                sys.stdout.write("%i-%i " % (e_idx,f_idx))
            # f_idx = 0
            # sys.stdout.write("%i-%i " % (e_idx,f_idx))
            # e_idx = english_sent.index(e_word)+1
            
        sys.stdout.write("\n")
            
        
    

    

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("arg_1", help="describe arg_1")
    # parser.add_argument("arg_2", help="describe arg_2")
    # parser.add_argument("-optional_arg", default=default_value, type=int/"", help='optional_arg meant for some purpose')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()


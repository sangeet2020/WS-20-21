#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-10 16:35:59
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-11 20:22:19




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
    
    def __init__(self, english_words, foreign_words, sentence_pairs, t_dict={}):
        self.english_words = english_words
        self.foreign_words = foreign_words
        self.sentence_pairs = sentence_pairs
        self.t_dict=t_dict
    
    def initial_probablity(self):
        print("Obtaining inital probablity")
        self.t_dict = dict((k,1 / (len(self.foreign_words))) for k in product(self.english_words,self.foreign_words))
        
        # for word_pair in product(self.english_words,self.foreign_words):
        #     self.t_dict[word_pair] = 1 / (len(self.foreign_words))
        #     # if (sys.getsizeof(self.t_dict) > 1):
        print(sys.getsizeof(self.t_dict)/1e+9,"GB")
    
    def cond_prob_e_f(self, e_words, f_words): # P(e|f)
        le = len(e_words)
        lf = len(f_words)
        
        epsilon = 1 # Normalization constant
        
        N_factor = epsilon / ((lf+1)**le)
        product = 1
        
        for j in range(le):
            inner_sum = 0
            for i in range(le):
                # import pdb; pdb.set_trace()
                inner_sum += self.t_dict[(e_words[j], f_words[i])]
            
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
            # print(self.cond_prob_e_f(e_words, f_words))
        
        pp = 2**pp
        return pp

    def aligner(self, epochs):
        
        print("I am here")
        self.initial_probablity()
        print("Inital probablity obtained")
        for _ in range(epochs):
            # pp = self.perplexity()
            # print("Perplexity: {}".format(pp))
            
            # Initialize
            total = {}
            count = {}
            for f_word in self.foreign_words:
                total[f_word] = 0
                for e_word in self.english_words:
                    count[(e_word, f_word)] = 0
            
            for sp in self.sentence_pairs:
                f_words = sp[0]
                e_words = sp[1]
                
                # Compute normalizations
                s_total = dict()
                for e in e_words:
                    s_total[e] = 0
                    for f in f_words:
                        s_total[e] += self.t_dict[(e, f)]
                
                # Collect counts
                for e in e_words:
                    for f in f_words:
                        count[(e, f)] += self.t_dict[(e, f)] / s_total[e]
                        total[f] += self.t_dict[(e, f)] / s_total[e]
            
            # Estimate probablities
            for f_word in self.foreign_words:
                for e_word in self.english_words:
                    self.t_dict[(e_word, f_word)] = count[(e_word, f_word)] / total[f_word]
            print("T-dict:")
            for i,v in self.t_dict.items():
                print(i,v)
            print("##"*20)
    
    def extractor(self):
        # Extracting alignments from probablity scores
        threshold=0.01
        alignments = {}
        for e_word in self.english_words:
            sig_score = {}
            for word_pair, score in self.t_dict.items():
                if word_pair[0] == e_word:
                    sig_score[word_pair[1]] = score
            # import pdb; pdb.set_trace()
            threshold_query = dict((k, v) for k, v in sig_score.items() if v >= threshold)
            alignments[e_word] = [*threshold_query.keys()]
            # import pdb; pdb.set_trace()
        # print(alignments)
        # print("Prob dict: ", self.t_dict)
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
    sentence_pairs = [ 
        [ ['das', 'Haus'], ['the', 'house'] ], 
        [ ['das', 'Buch'], ['the', 'book'] ], 
        [ ['ein', 'Buch'], ['a', 'book'] ]
    ]
    
    # e_file = "jhu-mt-hw/hw2/data/hansards.e"
    # f_file = "jhu-mt-hw/hw2/data/hansards.f"
    # e_file = "hansards.e"
    # f_file = "hansards.f"
    # sentence_pairs = []
    # start = time.time()
    # with open(f_file, 'r') as f_fpw, open(e_file, 'r') as e_fpw:
    #     file_iter = list(zip(f_fpw, e_fpw))
    #     for line in file_iter:
    #         f_tokens, e_tokens = tokenize(line)
    #         sentence_pairs.append([f_tokens, e_tokens])    
    # end=  time.time()
    # print("Total runtime data loading: %.3f s" % (end-start))
    
    # sentence_pairs = [ 
    #     [ ['la', 'maison'], ['the', 'house'] ], 
    #     [ ['la', 'maison', 'bleu'], ['the', 'blue', 'house'] ], 
    #     [ ['la', 'fleur'], ['the', 'flower'] ]
    # ]
    
    # import pdb; pdb.set_trace()
    print("No. of sentence of pairs= ", len(sentence_pairs))
    print("*"*40)
    print("Sentence pairs:")
    for pair in sentence_pairs:
        print(pair)
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
    ibm_model.aligner(epochs=7)
    end = time.time()
    print("Total runtime aligner: %.3f s" % (end-start))
    
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


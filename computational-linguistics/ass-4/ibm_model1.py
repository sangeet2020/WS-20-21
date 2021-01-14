#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-14 02:03:56
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-14 03:47:56

import math
import time
import os, sys
import argparse
import operator
import numpy as np
from tqdm import tqdm
from collections import defaultdict 


class IBM_model(object):
    
    def __init__(self, english_words, foreign_words, sentence_pairs, alpha=None, save_model=False):
        self.english_words = english_words
        self.foreign_words = foreign_words
        self.sentence_pairs = sentence_pairs
        self.alpha = alpha
        self.save_model = save_model
    
    def initial_probablity(self):
        print("Inital probablity...")
        self.t_score = np.full((len(self.english_words), 
                                len(self.foreign_words)), 
                                1 / (len(self.foreign_words)))
    
    def cond_prob_e_f(self, e_words, f_words): # P(e|f)
        le = len(e_words)
        lf = len(f_words)
        
        epsilon = 1 # Normalization constant
        
        N_factor = epsilon / ((lf)**le)
        product = 1
        for j in range(le):
            inner_sum = 0
            for i in range(lf):
                try:
                    e_idx, f_idx = (self.e_dict[e_words[j]], self.f_dict[f_words[i]])
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
            pp += -math.log(self.cond_prob_e_f(e_words, f_words), 2)
        # pp = 2**pp
        pp = np.power(2,pp)
        return pp

    def aligner(self, epochs):
        self.initial_probablity()
        print("\nEM algorithm...")
        
        self.e_dict = {k: v for v, k in enumerate(self.english_words)}
        self.f_dict = {k: v for v, k in enumerate(self.foreign_words)}
        ppxty = []
        for _ in tqdm(range(epochs), desc="Epochs   "):
            # pp = self.perplexity()
            # ppxty.append(pp)
            # print("Perplexity: {}".format(pp))
            
            # Initialize
            count = np.zeros((len(self.english_words), len(self.foreign_words)))
            total = np.zeros(len(self.foreign_words))
            
            # E- Step
            for sp in tqdm(self.sentence_pairs, desc="Sentences", total = len(self.sentence_pairs), leave=False):
                f_words = sp[0]
                e_words = sp[1]
                
                # Compute normalizations
                s_total = np.zeros(len(e_words))
                for eid, e in enumerate(e_words):
                    for _, f in enumerate(f_words):
                        e_idx, f_idx = (self.e_dict[e], self.f_dict[f])
                        s_total[eid] += self.t_score[(e_idx, f_idx)]
                        
                # Collect counts
                for eid, e in enumerate(e_words):
                    for _, f in enumerate(f_words):
                        e_idx, f_idx = (self.e_dict[e], self.f_dict[f])
                        count[(e_idx, f_idx)] += self.t_score[(e_idx, f_idx)] / s_total[eid]
                        total[f_idx] += self.t_score[(e_idx, f_idx)] / s_total[eid]
            
            # Estimate probablities: M-step
            self.t_score = count / total
            
        if self.save_model:
            print("Saving Model...")
            if self.alpha:
                outfile = "trans_prob_alpha_" + str(self.alpha) + "_epochs_" + str(epochs)+".npy"
            else:
                outfile = "trans_prob_bs" + "_epochs_" + str(epochs)+".npy"
            np.save(outfile, self.t_score)
            print("Model saved as", outfile)
            print("*"*30)
        # return ppxty
            
    def extractor(self):
        # Extracting alignments from probablity scores
        # self.t_score = np.load('trans_prob_alpha_0.35_epochs_10.npy')
        
        e_dict = {v: k for v, k in enumerate(self.english_words)}
        f_dict = {v: k for v, k in enumerate(self.foreign_words)}
        alignments = defaultdict(list)
        
        for eid, line in enumerate(self.t_score):
            if self.alpha:
                top_idx = np.argwhere(line >= self.alpha).T[0]
            else:
                top_idx = [np.argmax(line)]
            for _, fid in enumerate(top_idx): 
                alignments[e_dict[eid]].append(f_dict[fid])
        return alignments
    
    def _return_index(self, e_word, f_word):
        # Return the index in the form of tuple (e_idx, f_idx)
        # so as to append the score in the t_score matrix
        
        return (self.english_words.index(e_word), self.foreign_words.index(f_word))
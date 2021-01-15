#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-14 02:03:56
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-14 23:20:04

import math
import time
import os, sys
import argparse
import operator
import numpy as np
from tqdm import tqdm
from collections import defaultdict


class IBM_model(object):
    """ IBM Model-1

    Attributes:
        english_words   (list): list of all the unique words from source (English) text
        foreign_words   (list): list of all the unique words from target (foreign) text
        sentence_pairs  (list): list of all sentence pairs 
        alpha           (float, optional): threshold score of translation probability for alignment
        save_model      (bool, optional): choice to save the trained model
        
    """
    def __init__(self,
                 english_words,
                 foreign_words,
                 sentence_pairs,
                 alpha=None,
                 save_model=False):
        self.english_words = english_words
        self.foreign_words = foreign_words
        self.sentence_pairs = sentence_pairs
        self.alpha = alpha
        self.save_model = save_model

    def initial_probablity(self):
        """Uniformly initialize word translation probablities in a matrix
        where rows are indices of source (english) tokens and cols are 
        indices of target (foreign) tokens
        """
        print("Inital probablity...")
        self.t_score = np.full(
            (len(self.english_words), len(self.foreign_words)),
            1 / (len(self.foreign_words)))

    def aligner(self, epochs):
        """ Compute the translation probabilities using IBM Model-1.
        It uses only lexical translations and ignores any position information,
        resulting in translating multisets of words into multisets of words.

        Args:
            epochs (int): number of epochs
        """
        self.initial_probablity()
        print("\nEM algorithm...")

        self.e_dict = {k: v for v, k in enumerate(self.english_words)}
        self.f_dict = {k: v for v, k in enumerate(self.foreign_words)}
        # ppxty = []
        for _ in tqdm(range(epochs), desc="Epochs   "):
            # pp = self.perplexity()
            # ppxty.append(pp)
            # print("Perplexity: {}".format(pp))

            # Initialize
            count = np.zeros(
                (len(self.english_words), len(self.foreign_words)))
            total = np.zeros(len(self.foreign_words))

            # E- Step
            for sp in tqdm(self.sentence_pairs,
                           desc="Sentences",
                           total=len(self.sentence_pairs),
                           leave=False):
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
                        count[(e_idx,
                               f_idx)] += self.t_score[(e_idx,
                                                        f_idx)] / s_total[eid]
                        total[f_idx] += self.t_score[(e_idx,
                                                      f_idx)] / s_total[eid]

            # Estimate probablities: M-step
            self.t_score = count / total

        if self.save_model:
            print("Saving Model...")
            if self.alpha:
                outfile = "trans_prob_alpha_" + str(
                    self.alpha) + "_epochs_" + str(epochs) + ".npy"
            else:
                outfile = "trans_prob_bs" + "_epochs_" + str(epochs) + ".npy"
            np.save(outfile, self.t_score)
            print("Model saved as", outfile)
            print("*" * 30)
        # return ppxty

    def extractor(self):
        """Extract alignments from the translation probabilities computed above. 
        The goal is to get all possible translations for each source token.

        Returns:
            dict: a dictionary where each source (English) token is matched with
            all possible target (foreign) tokens.
        """
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

    def _perplexity(self):
        """NOT IN USE YET
        Compute perplexity to understand how well does the model fit the data.
        Reference (slide 34): http://mt-class.org/jhu/slides/lecture-ibm-model1.pdf 

        Returns:
            float: perplexity
        """
        pp = 0
        for pair in self.sentence_pairs:
            f_words = pair[0]
            e_words = pair[1]
            pp += -math.log(self._cond_prob_e_f(e_words, f_words), 2)
        # pp = 2**pp
        pp = np.power(2, pp)
        return pp

    def _cond_prob_e_f(self, e_words, f_words):
        """Compute conditional probablity i.e. P(e|f)
        i.e. probablity of target sentence given the source sentence.
        The formula is given by: [epsilon/ (lf+1)^le]*PI(Summation(t(e|f)))

        Args:
            e_words ([type]): list of source (english) tokens
            f_words ([type]): list of target (foreign) tokens

        Returns:
            float: conditional probablity P(e|f)
        """
        le = len(e_words)
        lf = len(f_words)

        epsilon = 1  # Normalization constant

        N_factor = epsilon / ((lf)**le)
        product = 1
        for j in range(le):
            inner_sum = 0
            for i in range(lf):
                e_idx, f_idx = (self.e_dict[e_words[j]],
                                self.f_dict[f_words[i]])
                inner_sum += self.t_score[(e_idx, f_idx)]
            product = inner_sum * product

        p_e_f = N_factor * product
        return p_e_f

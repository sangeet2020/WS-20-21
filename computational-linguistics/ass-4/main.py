#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-10 16:35:59
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-14 23:19:57
"""
Implementation of IBM Model 1, which is used in statistical 
machine translation to train an alignment model.
"""

import os
import time
import argparse
import numpy as np
from collections import defaultdict
from ibm_model1 import IBM_model
import matplotlib.pyplot as plt


def data_loader(e_file, f_file, args):
    """Load the source and target text file into a list with and also
    extract lists of unique tokens.

    Args:
        e_file (str): path to source (eng) file
        f_file (str): path to target (foreign) file

    Returns:
        tuple: comprising of a list of sentence pairs, list of tokens from 
        source text and list tokens from target text.
    """
    #**************** Some Toy Examples ****************
    # sentence_pairs = [
    #     [ ['la', 'maison'], ['the', 'house'] ],
    #     [ ['la', 'maison', 'bleu'], ['the', 'blue', 'house'] ],
    #     [ ['la', 'fleur'], ['the', 'flower'] ]
    # ]
    # sentence_pairs = [
    #     [ ['das', 'Haus'], ['the', 'house'] ],
    #     [ ['das', 'Buch'], ['the', 'book'] ],
    #     [ ['ein', 'Buch'], ['a', 'book'] ]
    # ]
    #************************************************
    sentence_pairs = [[sentence.strip().split() for sentence in pair]
                      for pair in zip(open(f_file), open(e_file))
                      ][:args.num_sents]
    english_words = []
    foreign_words = []
    for pair in sentence_pairs:
        for e_word in pair[1]:
            english_words.append(e_word)
        for f_word in pair[0]:
            foreign_words.append(f_word)
    english_words = sorted(list(set(english_words)), key=lambda s: s.lower())
    foreign_words = sorted(list(set(foreign_words)), key=lambda s: s.lower())
    return sentence_pairs, english_words, foreign_words


def perplexity_plot(ppxty, args):
    """PLot a perllexity vs iteration plot, given list of 
    perplexity values obtained after each iteration. 

    Args:
        ppxty (list): list of perplexity values
    """
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, args.epochs + 1), np.log10(ppxty), linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Log perplexity')
    plt.title('Log(2) perplexity vs. iterations')
    plt.show()


def main():
    """ main method """
    args = parse_arguments()
    net_start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    start = time.time()
    sentence_pairs, english_words, foreign_words = data_loader(
        args.eng_f, args.foreign_f, args)
    end = time.time()
    print("Total sentence pairs:", len(sentence_pairs))
    print("*" * 30)

    save_model = args.save_model
    if args.alpha is not None:
        alpha = args.alpha
    else:
        alpha = None
    ibm_model = IBM_model(english_words, foreign_words, sentence_pairs, alpha,
                          save_model)

    start = time.time()
    # ppxty = ibm_model.aligner(epochs=args.epochs)
    ibm_model.aligner(epochs=args.epochs)
    end = time.time()
    print("\naligner runtime: %.3f s" % (end - start))

    start = time.time()
    alignments = ibm_model.extractor()
    end = time.time()
    print("extractor runtime : %.3f s" % (end - start))

    outfile = str(args.out_dir) + "/ibm1.a"
    myfile = open(outfile, 'w')
    for sp in sentence_pairs:
        foreign_sent = sp[0]
        english_sent = sp[1]
        for e_word in english_sent:
            e_idx = english_sent.index(e_word)
            get_f_words = alignments[e_word]
            for word in get_f_words:
                if word in foreign_sent:
                    f_idx = foreign_sent.index(word)
                    myfile.write("%i-%i " % (f_idx, e_idx))
            # if args.use_null:
            #     if len(get_f_words) == 0:
            #         f_idx = "NULL"
            #         myfile.write("%i-%i " % (f_idx, e_idx))
        myfile.write("\n")
    myfile.close()
    net_end = time.time()
    print("Total runtime: %.3f s" % (net_end - net_start))
    # perplexity_plot(ppxty, args)
            
        
def parse_arguments():
    """ parse arguments """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eng_f", help="path to source (eng) file")
    parser.add_argument("foreign_f", help="path to target (foreign) file ")
    parser.add_argument("out_dir", help='output dir to save the obtained alignments')
    parser.add_argument("-epochs", default=10, type=int, help='number of training epochs for EM')
    parser.add_argument("-num_sents", default=10**6, type=int, help='number of sentences to train from')
    parser.add_argument("-alpha", default=None, type=float, help='threshold score of translation probability for alignment')
    parser.add_argument("-save_model", default=False, type=bool, help='save trained model')
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()
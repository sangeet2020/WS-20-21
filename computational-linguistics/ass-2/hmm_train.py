#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes


import os
import time
import argparse
from utils import *
from viterbi import *
from nltk.corpus.reader import TaggedCorpusReader

"""
A Bigram part-of-speech (POS) tagger based on supervised-Hidden Markov Models that utilizes Viterbi tagging.
"""


def main():
    """main function
    """
    n = 2  # Bigram HMM
    args = parse_arguments()
    treebank = TaggedCorpusReader(os.path.split(args.train_f)[0],
                                  os.path.split(args.train_f)[1])
    observation_space = [item[0] for item in treebank.sents()]  # all words
    state_space = [item[1] for item in treebank.sents()]  # all pos tags

    words = dict.fromkeys(observation_space)
    tags = dict.fromkeys(state_space)

    # HMM parameter estimation- initial, transition and emission probablity
    start = time.time()
    init_p = [item[1] for item in comp_initial(tags, treebank)]
    trans_p = comp_transition(n, tags, state_space)
    emission_p = comp_emission(
        words, tags, state_space, treebank, smoothing=args.smoothing)
    end = time.time()
    print("Runtime (training): %.3f s" %(end-start))

    # Test your HMM-trained model
    treebank = TaggedCorpusReader(os.path.split(args.eval_f)[0],
                                  os.path.split(args.eval_f)[1])
    viterbi_tags = []

    for sentence in treebank.paras():
        start = time.time()
        test_words = [item[0] for item in sentence]
        O, S, Y, pi, A, B = pre_process(
            words, tags, test_words, init_p, trans_p, emission_p)
        # Computes Viterbi's most likely tags
        
        if args.log_prob:
            X = viterbi_log(O, S, Y, pi, A, B)
        else:
            X = viterbi(O, S, Y, pi, A, B)    
        viterbi_tags.append(X)
    end = time.time()
    
    print("Runtime (viterbi): %.3f s" % (end - start))
    output_path = "./" + "de-tagger.tt"
    post_processing(viterbi_tags, args.test_f, output_path)


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train_f", help="path to train corpora")
    parser.add_argument("eval_f", help="path to eval file")
    parser.add_argument("test_f", help="path to test file")
    parser.add_argument("-smoothing",default=None,choices=['Laplace'], \
                        type=str,help='Smoothing techniques')
    parser.add_argument("-log_prob",default=None,choices=['log'], \
                        type=str,help='Viterbi Algorithm using log probabilities')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

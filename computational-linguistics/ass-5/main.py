#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-01-25 18:39:24
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-01-28 20:26:01

"""
Implement a Gibbs sampler which resamples a topic for each word in the corpus according
to the probability distribution in formula [5] (Griffiths & Steyvers 2004)
References:
- https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf?__=
- https://u.cs.biu.ac.il/~89-680/darling-lda.pdf
"""

import os
import sys
import pdb
import time
import logging
import argparse
import numpy as np
from itertools import chain
from collections import defaultdict


class LDA_Gibbs(object):
    """LDA Gibbs Sampling

    Attributes:
        alpha           (float) : Parameter that sets the topic distribution for the documents (default=0.02)
        beta            (float) : Parameter that sets the topic distribution for the words (default=0.1)
        num_topics      (int)   : number of topics (default=20)
        epochs          (int)   : number of training iterations (default=500)
        num_top_words   (int)   : number of words to select from each topic
        vocab_id_dict   (dict)  : dictionary mapped with word to its id
        docs            (list)  : nested list of tokenized words represented as int id
        
    """
    def __init__(self, alpha, beta, num_topics, epochs, num_top_words,
                vocab_id_dict, docs):
        self.alpha = alpha
        self.beta = beta
        self.num_topics = num_topics
        self.epochs = epochs
        self.vocab_id_dict = vocab_id_dict
        self.docs = docs
        self.num_top_words = num_top_words

    def random_initialization(self):
        """ The goal is to randomly assign topics to each word in each document.
        Then we calculate a word to topic count matrix (self.wt) and
        a document to topic count matrix (self.dt).
        """

        start = time.time()
        self.num_words = len(self.vocab_id_dict)
        self.num_docs = len(self.docs)

        # topic-word distribution matrix: Topics X Words
        self.wt = np.zeros(shape=(self.num_topics, self.num_words))

        # document-topic distribution: Docs X Topics
        self.dt = np.zeros(shape=(self.num_docs, self.num_topics))

        self.t = np.zeros(self.num_topics)
        self.d = np.zeros(self.num_docs)
        self.topics = defaultdict(int)

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
        """Given a word in a document, compute the probability of all topics.
        This is computed using collapsed Gibbs Sampling formula.

        Args:
            w (int): word
            d (int): document id

        Returns:
            [ndarray]: normalized conditional probablity for all topics
        """

        prob_z = (self.wt[:, w] + self.beta) / (self.t + (self.num_words * self.beta)) * \
            (self.dt[d, :] + self.alpha) / (self.d[d] + (self.num_topics * self.alpha))
        prob_z /= np.sum(prob_z)
        return prob_z

    def train(self):
        """Run the Gibbs sampler. The goal is to loop over the desired number of iterations 
        wherein each loop a topic is sampled for each word instance in the corpus. Steps:
        (i) randomly assign topics to each word in each document
        (ii) decrement the counts associated with the current assignment. This is done because 
        the Gibbs sampling procedure involves sampling from distributions conditioned on 
        all other variables, so we must remove the current assignment.
        (iii) calculate the probability of each topic assignment (eqn 5) (Griffiths & Steyvers 2004)
        (iv) sample the distribution computed above and the chosen topic is set in self.topics
        (v)  increment the counts again
        Once the training process the complete, extract the frequent occurring words for each topic
        
        Returns:
            [list]: list of frequent words for each topic
        """

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
                    z = np.random.choice(np.arange(self.num_topics), p=prob_z)
                    self.topics[(doc_id, word_id)] = z
                    self.dt[doc_id, z] += 1
                    self.d[doc_id] += 1
                    self.wt[z, word] += 1
                    self.t[z] += 1
            end = time.time()
            logging.info("Epoch: {}/{}, runtime: {:0.3f} s".format(
                epoch + 1, self.epochs, end - start))

        word_list = self._get_topicwise_words()
        return word_list

    def _get_topicwise_words(self):
        """Given the word to topic distribution matrix, sort the matrix over topic axis
        and get the list of most frequently occurring words for each topic.

        Returns:
            [list]: frequent words for each topic
        """

        self.wt_sorted = (-self.wt).argsort()[:, :self.num_top_words]

        self.vocab_id_dict = {v: k for k, v in self.vocab_id_dict.items()}
        word_list = [
            self._idx2word(self.wt_sorted[topic_id])
            for topic_id in range(self.num_topics)
        ]
        return word_list

    def _idx2word(self, idx_list):
        """ convert list of word ids into list of words as mapped in `vocab_id_dict`
        dictionary.

        Args:
            idx_list (list): list of word ids

        Returns:
            list: list of words
        """

        return [self.vocab_id_dict[x] for x in idx_list]


def data_loader(args):
    """ Tokenize the given corpus, and assign word IDs to each unique word
    in all documents.

    Returns:
        tuple: a tuple of vocab2id dictornary and nested list of tokenized 
        (split into words) documents 
    """

    start = time.time()
    doc_list = [line.split()
                for line in open(args.corpus_f)][1:][:args.num_sents]
    vocab = list(chain.from_iterable(doc_list))

    vocab_id_dict = {}
    idx = 0
    for word in vocab:
        if word not in vocab_id_dict:
            vocab_id_dict[word] = idx
            idx += 1
    docs = []
    for doc in doc_list:
        temp = [vocab_id_dict[word] for word in doc]
        docs.append(np.asarray(temp))
    end = time.time()
    logging.info("\nData loading runtime: %.3f s" % (end - start))
    return vocab_id_dict, docs


def main():
    """ main method """
    args = parse_arguments()
    start = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s %(message)s',
                        filename=args.out_dir + '/' + 'training.log',
                        filemode='a',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Expected runtime: 3.3 hrs")

    vocab_id_dict, docs = data_loader(args)
    lda_gibbs = LDA_Gibbs(args.alpha, args.beta, args.num_topics, args.epochs,
                        args.num_top_words, vocab_id_dict, docs)
    word_list = lda_gibbs.train()

    outfile = str(args.out_dir) + "/topicwise_words.txt"
    myfile = open(outfile, 'w')
    for idx, words in enumerate(word_list):
        words = ', '.join(map(str, words))
        myfile.write("Topic %i: %s. " % (idx + 1, words))
        myfile.write("\n")
    myfile.close()
    end = time.time()
    tot_rt = (end - start) / 3600
    logging.info('%s %.3f %s', 'Total runtime:', tot_rt, 'hrs.')
    logging.info('%s: %s%s', 'Results saved to:', args.out_dir,
                '/topicwise_words.txt')
    logging.info('%s', '-- Done --')


def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus_f", help="path to input text corpus file")
    parser.add_argument("out_dir",
                        help="path to save the frequent words for each topic")
    parser.add_argument(
        "-alpha",
        default=0.02,
        type=float,
        help='Parameter that sets the topic distribution for the documents')
    parser.add_argument(
        "-beta",
        default=0.1,
        type=float,
        help='Parameter that sets the topic distribution for the words')
    parser.add_argument("-num_topics",
                        default=20,
                        type=int,
                        help='number of topics')
    parser.add_argument("-epochs",
                        default=500,
                        type=int,
                        help='number of training iterations')
    parser.add_argument("-num_sents",
                        default=2001,
                        type=int,
                        help='number of sentences to train from')
    parser.add_argument("-num_top_words",
                        default=10,
                        type=int,
                        help='number of words to select from each topic')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

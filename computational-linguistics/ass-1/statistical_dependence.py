#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes

"""
Calculate the pmi for all successive pairs (w1 , w2 ) of words in a corpus
pmi(w1 , w2) = log[(C(w1 w2)*N) / (C(w1)*C(w2))]
"""

import nltk
import math
import string
import operator
import itertools
import collections
from nltk.util import ngrams
import matplotlib.pyplot as plt


def data_prep(filename):
    """Perform pre-processing steps in the input file and tokenize it.

    Args:
        filename (str): path to file

    Returns:
        list:tokens- list containing tokenized words of the input text file 
    """

    file_content = open(filename, 'r', encoding='utf-8-sig').read()
    file_content = file_content.lower()
    # Strip punctuations. Reference: https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
    file_content = file_content.translate(
        str.maketrans('', '', string.punctuation))
    tokens_list = nltk.word_tokenize(file_content)
    # Remove tokens with frequnecy less than 10
    tokens_list = [item for item in tokens_list if collections.Counter(tokens_list)[
        item] >= 10]

    return tokens_list


def compute_pmi(word_pair, N, global_dict):
    """Givem word-pairs and tokens from the corpora, compute the PMI score

    Args:
        word_pair (tuple): tuple of word pairs- (w1, w2)
        N (list): length of corpors
        global_dict (dict): dict consisting frequency of each tokens and token_pairs

    Returns:
        float: PMI score of the given word-pair
    """

    counts = lookup(word_pair, global_dict)
    return math.log(((counts[2] * N))/(counts[0] * counts[1]))


def lookup(word_pair, global_dict):
    """Compute counts of each word in the word pair and the word pair itself in the list of all token pairs

    Args:
        word_pair (tuple): tuple of word pairs- (w1, w2)
        global_dict (dict): dict consisting frequency of each tokens and token_pairs

    Returns:
        list: list containing counts of w1, w2 (in the tokens list) and counts of word_pair (in the token_pairs list)
    """

    Cw1 = global_dict.get(word_pair[0])
    Cw2 = global_dict.get(word_pair[1])

    Cw1_w2 = global_dict.get(word_pair)
    return [Cw1, Cw2, Cw1_w2]


def print_scores(sort_tok_dict, l, rev=False):
    """Print PMI scores in a tabulated format

    Args:
        sort_tok_dict (dict): dictionary containing word-pairs are key and PMI scores as values
        l (int): maximum word-pairs for which PMI scores have to be printed
        rev (bool): Choice to reverse the dict. Defaults to False.
    """
    # References: https://www.geeksforgeeks.org/python-get-first-n-keyvalue-pairs-in-given-dictionary/

    if rev:
        out = dict(itertools.islice(sort_tok_dict.items(),
                                    len(sort_tok_dict)-l, len(sort_tok_dict)))
        out = dict(sorted(out.items(), key=operator.itemgetter(1), reverse=False))
    else:
        out = dict(itertools.islice(sort_tok_dict.items(), l))
    dash = '-' * 32
    print(dash)
    print('{:<10s}{:>10s}{:>12s}'.format("w1", "w2", "pmi"))
    print(dash)
    for key, value in out.items():
        print('{:<10s}{:>10s}{:>12s}'.format(
            key[0], key[1], str(format(value, ".3f"))))


if __name__ == "__main__":
    """main function"""

    filename = "data/junglebook.txt"
    tokens = data_prep(filename)
    N = len(tokens)
    token_pairs = list(ngrams(tokens, 2))

    # Get a dict with combined counts of unigrams and bigrams
    global_dict = nltk.FreqDist(tokens + token_pairs)

    # create a dict with keys= word pairs, and value= None
    tok_dict = dict.fromkeys(token_pairs)
    for word_pair, pmi_score in tok_dict.items():
        pmi_score = compute_pmi(word_pair, N, global_dict)
        tok_dict[word_pair] = pmi_score

    sort_tok_dict = dict(
        sorted(tok_dict.items(), key=operator.itemgetter(1), reverse=True))

    print_scores(sort_tok_dict, l=20)
    print_scores(sort_tok_dict, l=20, rev=True)

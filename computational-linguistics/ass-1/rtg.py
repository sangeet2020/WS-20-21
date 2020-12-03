#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes

"""
Given a text file, train a ngram instance to compute probablity disribution
and generate a random sentence.
"""

import nltk
from ngram import *
from nltk.corpus import gutenberg
from nltk.tokenize.treebank import TreebankWordDetokenizer


def data_prep(filename):
    """Perform pre-processing steps in the input file and tokenize it.

    Args:
        filename (str): path to file

    Returns:
        list:tokens- list containing tokenized words of the input text file 

    """
    file_content = open(filename, 'r', encoding='utf-8-sig').read()
    file_content = file_content.lower()
    tokens = nltk.word_tokenize(file_content)
    return tokens


def text_generate(tokens, n, length, estimator):
    """Train an ngram model to compute probablity disribution and generate a 
    sequence of random words.

    Args:
        tokens (str): list of tokens
        n (str): number of previous tokens taken into account to compute prob. 
        distribution of the next token
        length (str): maximum sequence length of randomly generated words
        estimator (str): ml_estimator or SimpleGoodTuringProbDist

    Returns:
        None

    """
    ngram = BasicNgram(n, tokens, estimator=estimator)
    start_seq = list(ngram.contexts()[0])
    # print(start_seq)
    sent = []
    for i in range(length):
        word = ngram[tuple(start_seq)].generate()
        start_seq = start_seq[1:] + [word]
        # print(start_seq)
        sent.append(word)
    post_process(sent)


def post_process(sent):
    """Post process randomly generated sequnece of words into readable sentence.

    Args:
        sent (list): sequence of random words

    Returns:
        None

    """
    # References: https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
    result = TreebankWordDetokenizer().detokenize(sent)
    # result = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sent)
    result = '. '.join(
        map(lambda s: s.strip().capitalize(), result.split('.')))

    print(result)


if __name__ == "__main__":
    """main function
    """
    filename = "data/kingjamesbible_tokenized.txt"
    tokens = data_prep(filename)
    punct_sent = text_generate(tokens, n=2, length=100, estimator=ml_estimator)
    punct_sent = text_generate(tokens, n=3, length=100, estimator=ml_estimator)
    punct_sent = text_generate(tokens, n=4, length=100, estimator=ml_estimator)
    punct_sent = text_generate(
        tokens, n=3, length=100, estimator=goodturing_estimator)
    punct_sent = text_generate(
        tokens, n=3, length=100, estimator=goodturing_estimator)

    # References: https://www.nltk.org/book/ch02.html
    emma = gutenberg.words('austen-emma.txt')[1:]
    punct_sent = text_generate(emma, n=2, length=100, estimator=ml_estimator)
    punct_sent = text_generate(emma, n=3, length=100, estimator=ml_estimator)

#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes


import time
import numpy as np
import collections

def create_ngram(n, tok_list):
    """Given a list and the ngram order, return ngrams

    Args:
        n (int): ngram order e.g. n=2
        tok_list (list): token list

    Returns:
        list: generate ngrams for the given list of tokens
    """
    
    return [tok_list[i:i+ n] for i in range(len(tok_list)-n+1)]


def comp_initial(tags, treebank):
    """Compute initial probability for the given set of tags and training corpus

    Args:
        tags (list): list of unique tags from the corpus
        treebank (object): File object with train corpora loaded. nltk.corpus.reader.tagged.TaggedCorpusReader

    Returns:
        list: list of tuples containing POS tag and its initial probability. Size: 1 X K
    """
    
    parsed_sents = treebank.paras()
    f_tags = [line[0][1] for line in parsed_sents]
    freq_f_tags = collections.Counter(f_tags)
    init_p = [freq_f_tags[item]/len(f_tags) for item in list(tags)]
    return list(zip(list(tags), init_p))

def comp_transition(n, tags, state_space):
    """Compute transition probablity matrix for the given set of tags.

    Args:
        n (int): ngram order.E.g. n=2 for bigrams
        tags (dict): a dictionary of all unique POS tags
        state_space (list): list of all tags in the train set

    Returns:
        [numpy.ndarray]: 2D array of transition probablities. Size= K X K
    """

    start = time.time()
    K = len(tags)
    ngram_state_space = create_ngram(n, state_space)
    trans_p = np.zeros(shape=(K, K))
    frq_tags = collections.Counter(state_space)

    for row in range(K):
        for col in range(K):
            t1 = list(tags)[row]
            t2 = list(tags)[col]
            C_t1 = frq_tags[t1]
            C_t1_t2 = 0
            for i in ngram_state_space:
                if i[0] == t1 and i[1] == t2:
                    C_t1_t2 += 1
            trans_p[row, col] = C_t1_t2 / C_t1
    
    end = time.time()
    print("Runtime (transition prob): %.3f s" %(end-start))
    return trans_p


def comp_emission(words, tags, state_space, treebank, smoothing=None):
    """Compute emission probablity matrix for the given set of tags and words in the training corpora.

    Args:
        words (dict): a dictionary of all unique words
        tags (dict): a dictionary of all unique POS tags
        state_space (list): list of all tags in the train set
        treebank (object): File object with train corpora loaded. nltk.corpus.reader.tagged.TaggedCorpusReader
        smoothing (str, optional): Smoothing technique. Defaults to None.

    Returns:
        [numpy.ndarray]: 2D array of emission probablities. Size= N X K
    """
    start = time.time()
    N = len(words)
    K = len(tags)
    emission_p = np.zeros(shape=(K,N))
    word_tag_pairs = [tuple(item) for item in treebank.sents()]
    frq_tags = collections.Counter(state_space)
    
    frq_word_tags = collections.Counter(word_tag_pairs)
    for row in range(K):
        for col in range(N):
            t1 = list(tags)[row]
            w1 = list(words)[col]
            C_t1_w1 = frq_word_tags[(w1, t1)]
            C_t1 = frq_tags[t1]
            if smoothing == 'Laplace':
                emission_p[row, col] = (C_t1_w1 + 1)/ (C_t1 + N)
            else:
                emission_p[row, col] = (C_t1_w1 / C_t1)
    end = time.time()
    print("Runtime (emission prob): %.3f s" %(end-start))
    return emission_p


def pre_process(words, tags, test_tagged_words, init_p, trans_p, emission_p):
    """Preapre input arguments in an acceptable format for Viterbi algo and handle unkown
    tokens.

    Args:
        words (dict): a dictionary of all unique words
        tags (dict): a dictionary of all unique POS tags
        test_tagged_words (list): input sequence of words.
        init_p (numpy.ndarray): list of probablity. Size: 1 X K
        trans_p (numpy.ndarray):  2D array of transition probablities. Size= K X K
        emission_p (numpy.ndarray): 2D array of emission probablities. Size= N X K

    Returns:
        [tuple]: ready to pass argumnets for viterbi algo
    """
    O = np.arange(0,len(words))
    S = np.asarray(list(tags))
    
    # Y = [list(words).index(item) for item in test_tagged_words]
    Y = []
    for item in test_tagged_words:
        # Unknown word handling. If test word in not found in train set, represent it by -1.
        # Later, when iterating through this index we set B[i, Y[j]] = 1 for all tags.
        if item in list(words):
            Y.append(list(words).index(item))
        else:
            Y.append(-1)
    
    pi = init_p
    A = trans_p
    B = emission_p
    
    return O,S,Y, pi, A, B

def post_processing(viterbi_tags, test_file, output_file):
    """ Save the input sequence as first column and predicted POS tags as second column in CoNLL format.

    Args:
        viterbi_tags (list): list containing predicted viterbi POS tags
        test_file (str): path to test file with no-POS tags
        output_file (str): output path to save POS tagged file in CoNNL format
    """

    viterbi_tags_flat = [item for sublist in viterbi_tags for item in sublist]
    with open(test_file) as ifh, open(output_file, 'w') as ofh:
        i = 0
        for line in ifh:
            word = line.rstrip()                 # remove newline
            if not word.strip():
                # Empty line detected. Insert a blank line to identify a new sentence.
                ofh.write('\n')
            else:
                tag = viterbi_tags_flat[i]
                ofh.write(word+"\t"+tag+'\n')  # write line
                i += 1
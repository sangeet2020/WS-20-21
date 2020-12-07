import pdb
import time
import nltk
import collections
import numpy as np
import pandas as pd
from viterbi import viterbi
from nltk.corpus.reader import TaggedCorpusReader

# np.set_printoptions(formatter={'float': "{: 7.4f}".format})

def create_ngram(n, tok_list):
    # Given a list and the ngram order, return ngrams
    
    return [tok_list[i:i+ n] for i in range(len(tok_list)-n+1)]

def comp_initial(tags, treebank):
    parsed_sents = treebank.paras()
    f_tags = [line[0][1] for line in parsed_sents]
    freq_f_tags = collections.Counter(f_tags)
    init_p = [freq_f_tags[item]/len(f_tags) for item in list(tags)]
    return list(zip(list(tags), init_p))

def comp_transition(n, tags, state_space):
    
    # tags = list of uniqiue POS tags
    # state_space = list of all tags from the tain data
    
    K = len(tags)
    ngram_state_space = create_ngram(n, state_space)
    trans_p = np.zeros(shape=(K, K)) # transition matrix
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
        
    return trans_p


def comp_emission(words, tags, state_space, treebank, smoothing=None):
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
                emission_p[row, col] = (C_t1_w1 / C_t1) # 0 can be added to leave no probs 0
    
    return emission_p


def post_process(words, tags, test_tagged_words, init_p, trans_p, emission_p, word_tag_pairs):
                #   O     S       Y                 pi      A       B
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
    X = viterbi(O,S,Y, pi, A, B)
    
    
    viterbi_tagged_pairs = list(zip(test_tagged_words, X))
    print("HMM-viterbi tagged pairs\n", viterbi_tagged_pairs[:20],'\n')
    
    with open('my_test.tt') as ifh, open('tagger.tt', 'w') as ofh:
        i = 0
        for line in ifh:
            word = line.rstrip()                 # remove newline
            if not word.strip():
                # Empty line detected. Insert a blank line to identify a new sentence.
                ofh.write('\n')
            else:
                tag = list(viterbi_tagged_pairs[i])[1]         
                ofh.write(word+"\t"+tag+'\n')  # write line
                i += 1

    
    print("Ground truth\n",word_tag_pairs[:20],'\n')
    print(len(viterbi_tagged_pairs), len(word_tag_pairs))
    check = [i for i, j in zip(viterbi_tagged_pairs, word_tag_pairs) if i == j] 
    accuracy = len(check)/len(viterbi_tagged_pairs)
    print('Viterbi Algorithm Accuracy: ',accuracy*100)
    
    
n = 2 # Bi-gram HMM
corpus_path = '.'
# corpus_files = 'de-train.tt'
corpus_files = 'my_train.tt'
treebank = TaggedCorpusReader(corpus_path, corpus_files)

observation_space = [item[0] for item in treebank.sents()] # all words
state_space = [item[1] for item in treebank.sents()] # all pos tags

# get unique words and tags
words = dict.fromkeys(observation_space)
tags = dict.fromkeys(state_space)

# Transition probablity matrix
start = time.time()
trans_p = comp_transition(n, tags, state_space)
tags_df = pd.DataFrame(trans_p, columns = list(tags), index=list(tags))
# print(tags_df)
end = time.time()
difference = end-start
print("Runtime (transition matrix): ", difference)

# Initial probablity
init_p = [item[1] for item in comp_initial(tags, treebank)]
# print(init_p)

# Emission probablity matrix- expensive runtime
start = time.time()
emission_p = comp_emission(words, tags, state_space, treebank, smoothing=None)
end = time.time()
difference = end-start
# print(emission_p.nbytes)
tags_df = pd.DataFrame(emission_p, columns = list(words), index=list(tags))
# print(tags_df)

print("Runtime (emission matrix): ", difference)

# Test your HMM-trained model
corpus_path = '.'
# corpus_files = 'de-eval.tt'
corpus_files = 'my_eval.tt'
treebank = TaggedCorpusReader(corpus_path, corpus_files)

test_words = [item[0] for item in treebank.sents()]
test_word_tag_pairs = [tuple(item) for item in treebank.sents()] # Ground truth

# Performs
    # computes Viterbi's most likely tags
    # perform some post processing i.e. appending Tags to the test file in CoNNL format.
X = post_process(words, tags, test_words, init_p, trans_p, emission_p, test_word_tag_pairs) 
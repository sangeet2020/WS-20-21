import pdb
from nltk.corpus.reader.knbc import test
import numpy as np
import nltk
from nltk.corpus.reader import TaggedCorpusReader
import collections
import pandas as pd
import time

np.set_printoptions(formatter={'float': "{: 7.4f}".format})

def create_ngram(n, tok_list):
    # Given a list and the ngram order, return ngrams
    
    return [tok_list[i:i+ n] for i in range(len(tok_list)-n+1)]

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

def comp_emission(words, tags, state_space, treebank):
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
            emission_p[row, col] = C_t1_w1 / C_t1
    
    return emission_p

def viterbi(O,S,Y, pi, A, B):
    start = time.time()
    # O = np.arange(1,7) # observation space # all words # Size = 1 X N
    # S = np.asarray([0, 1, 2]) # State space # all POS tags # Size = 1 X K

    # Y = np.array([0, 2, 0, 2, 2, 1]).astype(np.int32) # Observation sequnece T # A sent that needs POS tags predictions 
    # # Size = 1 X T

    # pi = np.array([0.6, 0.2, 0.2]) # Initial probablity # Size = 1 X K

    # A = np.array([[0.8, 0.1, 0.1], 
    #             [0.2, 0.7, 0.1], 
    #             [0.1, 0.3, 0.6]]) # transition matrix # Size = K X K
    # B = np.array([[0.7, 0.0, 0.3], 
    #             [0.1, 0.9, 0.0], 
    #             [0.0, 0.2, 0.8]]) # emission matrix # Size = N X K
    # print("O",O)
    # print("S",S)
    # print("pi",pi)
    # print("Y",Y)
    # print("A",A,'\n')
    # print("B",B)

    N = len(O)
    K = len(S)
    T = len(Y)
    
    T1 = np.zeros(shape=(K,T))
    T2 = np.zeros(shape=(K,T))
    for i in range(K):
        T1[i,0] = pi[i] * B[i, Y[0]] # or 0 = i-1
        T2[i,0] = 0
        
    for j in range(1, T):    
        for i in range(K):
    #         print(j, i)
    #         print(B[i, Y[j]])
            if Y[j] == -1:
                # Unkown word handling. Set B[i, Y[j]] = 1 for all tags if Y[j] == -1 aka word not found in train set.
                next_prob = T1[:,j-1] * A[:, i] * 1
            else:    
                next_prob = T1[:,j-1] * A[:, i] * B[i, Y[j]]
            T1[i,j] = np.max(next_prob)
            T2[i,j] = np.argmax(next_prob)
    #         print(T1,'\n')
        
    # import pdb; pdb.set_trace()
    Z = [None] * T
    X = [None] * T

    # Backpointer
    Z[T-1] = np.argmax(T1[:,T-1])
    X[T-1] = S[Z[T-1]]

    for j in reversed(range(1, T)):
        Z[j-1] = T2[int(Z[j]),j]
        X[j-1] = S[int(Z[j-1])]
    print('***************')
    end = time.time()
    print("Viterbi run time: %.3f" %(end-start))
    
    return X

def pre_process(words, tags, test_tagged_words, init_p, trans_p, emission_p, word_tag_pairs):
            #   O       S       Y                   pi      A       B
    O = np.arange(1,len(words))
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
    # print("O",O)
    # print("S",S)
    # print("Y",Y)
    # print("Word tag pairs", word_tag_pairs)
    # print("pi",pi)
    # print("A",A,'\n')
    # print("B",B)
    X = viterbi(O,S,Y, pi, A, B)
    # print(X)
    # print(word_tag_pairs)
    
    
    viterbi_tagged_pairs = list(zip(test_tagged_words, X))
    print(viterbi_tagged_pairs[:],'\n')
    print(word_tag_pairs[:],'\n')
    print(len(viterbi_tagged_pairs), len(word_tag_pairs))
    check = [i for i, j in zip(viterbi_tagged_pairs, word_tag_pairs) if i == j] 
    accuracy = len(check)/len(viterbi_tagged_pairs)
    print('Viterbi Algorithm Accuracy: ',accuracy*100)
    
    
n = 2 # Bi-gram HMM
corpus_path = '.'
corpus_files = 'de-eval.tt'
# corpus_files = 'test.txt'
treebank = TaggedCorpusReader(corpus_path, corpus_files)

observation_space = [item[0] for item in treebank.sents()] # all words
state_space = [item[1] for item in treebank.sents()] # all pos tags

# get unique words and tags
words = set(observation_space)
tags = set(state_space)

start = time.time()
trans_p = comp_transition(n, tags, state_space)
tags_df = pd.DataFrame(trans_p, columns = list(tags), index=list(tags))
# print(tags_df)

end = time.time()
difference = end-start
print("Time taken in seconds: ", difference)

start = time.time()
emission_p = comp_emission(words, tags, state_space, treebank)
end = time.time()
difference = end-start
print(emission_p.nbytes)
tags_df = pd.DataFrame(emission_p, columns = list(words), index=list(tags))
# print(tags_df)
print("Time taken in seconds: ", difference)

# print(tagged_sents,'\n')
# print(word_tag_pairs,'\n')



corpus_path = '.'
corpus_files = 'test.txt'
treebank = TaggedCorpusReader(corpus_path, corpus_files)

test_words = [item[0] for item in treebank.sents()]
# test_tags = [item[1] for item in treebank.sents()] # all pos tags

test_word_tag_pairs = [tuple(item) for item in treebank.sents()] # Ground truth


# test_tagged_words = observation_space
# print(test_tagged_words)

# word_tag_pairs = word_tag_pairs[:50]
init_p = np.ones(shape=(len(test_words))) * 0.5
# print(init_p)
# print(words)
X = pre_process(words, tags, test_words, init_p, trans_p, emission_p, test_word_tag_pairs)




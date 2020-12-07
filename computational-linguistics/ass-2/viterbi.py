#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes


import numpy as np

def viterbi(O,S,Y, pi, A, B):
    """Generates a path which is a sequence of most likely states that generates the given observation Y.

    Args:
        O (numpy.ndarray): observation space.      Size: 1 X N
        S (numpy.ndarray): state space.            Size: 1 X K
        Y (list): observation sequence.            Size: 1 X T
        pi (numpy.ndarray): inial probablities.    Size: 1 X K
        A (numpy.ndarray): transition matrix.      Size: K X K
        B (numpy.ndarray): emission matrix         Size: N X K

    Returns:
        list: list of most likely sequence of POS tags
    """
    # Reference: https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    #**************************************************************************
    ## Example data for trial
    # input
        # O = np.arange(1,7) # observation space # uniq words # Size = 1 X N
        # S = np.asarray([0, 1, 2]) # State space # uniq POS tags # Size = 1 X K

        # Y = np.array([0, 2, 0, 2, 2, 1]).astype(np.int32) # Observation sequnece T
        # # Size = 1 X T

        # pi = np.array([0.6, 0.2, 0.2]) # Initial probablity # Size = 1 X K

        # A = np.array([[0.8, 0.1, 0.1], 
        #             [0.2, 0.7, 0.1], 
        #             [0.1, 0.3, 0.6]]) # transition matrix # Size = K X K
        # B = np.array([[0.7, 0.0, 0.3], 
        #             [0.1, 0.9, 0.0], 
        #             [0.0, 0.2, 0.8]]) # emission matrix # Size = K X N
    
        # print("O",O)
        # print("S",S)
        # print("pi",pi)
        # print("Y",Y)
        # print("A",A,'\n')
        # print("B",B)
    
    # output
    #   X = [0, 0, 0, 2, 2, 1] # Most likely path/sequence

    #**************************************************************************

    N = len(O)
    K = len(S)
    T = len(Y)
    T1 = np.zeros(shape=(K,T))
    T2 = np.zeros(shape=(K,T))
    for i in range(K):
        T1[i,0] = pi[i] * B[i, Y[0]]
        T2[i,0] = 0
    
    
    for j in range(1, T):    
        for i in range(K):
            if Y[j] == -1:
                # Unkown word handling. Set B[i, Y[j]] = 1 for all tags if Y[j] == -1 
                # aka word not found in train set.
                next_prob = T1[:,j-1] * A[:, i] * 1
            else:    
                next_prob = T1[:,j-1] * A[:, i] * B[i, Y[j]]
            T1[i,j] = np.max(next_prob)
            T2[i,j] = np.argmax(next_prob)
    
    
    Z = [None] * T
    X = [None] * T

    # Backpointer
    Z[T-1] = np.argmax(T1[:,T-1])
    X[T-1] = S[Z[T-1]]
    
    for j in reversed(range(1, T)):
        Z[j-1] = T2[int(Z[j]),j]
        X[j-1] = S[int(Z[j-1])]
    
    return X # Most likely tags


def viterbi_log(O,S,Y, pi, A, B):
    """Generates a path which is a sequence of most likely states that generates the given observation Y.

    Args:
        O (numpy.ndarray): observation space.      Size: 1 X N
        S (numpy.ndarray): state space.            Size: 1 X K
        Y (list): observation sequence.            Size: 1 X T
        pi (numpy.ndarray): inial probablities.    Size: 1 X K
        A (numpy.ndarray): transition matrix.      Size: K X K
        B (numpy.ndarray): emission matrix         Size: N X K

    Returns:
        list: list of most likely sequence of POS tags
    """
    # Reference: https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
    #**************************************************************************
    ## Example data for trial
    # input
        # O = np.arange(1,7) # observation space # uniq words # Size = 1 X N
        # S = np.asarray([0, 1, 2]) # State space # uniq POS tags # Size = 1 X K

        # Y = np.array([0, 2, 0, 2, 2, 1]).astype(np.int32) # Observation sequnece T
        # # Size = 1 X T

        # pi = np.array([0.6, 0.2, 0.2]) # Initial probablity # Size = 1 X K

        # A = np.array([[0.8, 0.1, 0.1], 
        #             [0.2, 0.7, 0.1], 
        #             [0.1, 0.3, 0.6]]) # transition matrix # Size = K X K
        # B = np.array([[0.7, 0.0, 0.3], 
        #             [0.1, 0.9, 0.0], 
        #             [0.0, 0.2, 0.8]]) # emission matrix # Size = K X N
    
        # print("O",O)
        # print("S",S)
        # print("pi",pi)
        # print("Y",Y)
        # print("A",A,'\n')
        # print("B",B)
    
    # output
    #   X = [0, 0, 0, 2, 2, 1] # Most likely path/sequence

    #**************************************************************************

    tiny = np.finfo(0.).tiny #  limits for floating point types.
    N = len(O)
    K = len(S)
    T = len(Y)
    # pi = np.log(pi + tiny)
    # A = np.log(A + tiny)
    # B = np.log(B + tiny)
    
    T1 = np.zeros(shape=(K,T))
    T2 = np.zeros(shape=(K,T))
    for i in range(K):
        T1[i,0] = pi[i] * B[i, Y[0]]
        T2[i,0] = 0
    
    
    for j in range(1, T):    
        for i in range(K):
            if Y[j] == -1:
                # Unkown word handling. Set B[i, Y[j]] = 1 for all tags if Y[j] == -1 
                # aka word not found in train set.
                next_prob = T1[:,j-1] + np.log(A[:, i])
            else:    
                next_prob = T1[:,j-1] + np.log(A[:, i]) + np.log(B[i, Y[j]])
            T1[i,j] = np.max(next_prob)
            T2[i,j] = np.argmax(next_prob)
    
    
    Z = [None] * T
    X = [None] * T

    # Backpointer
    Z[T-1] = np.argmax(T1[:,T-1])
    X[T-1] = S[Z[T-1]]
    
    for j in reversed(range(1, T)):
        Z[j-1] = T2[int(Z[j]),j]
        X[j-1] = S[int(Z[j-1])]
    
    return X # Most likely tags
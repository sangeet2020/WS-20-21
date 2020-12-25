#!/usr/bin/python3
# -*- coding: utf-8 -*-

# author : Sangeet Sagar
# e-mail : sasa00001@stud.uni-saarland.de
# Organization: Universität des Saarlandes
# Date: Friday 25 December 2020 01:46:21 AM CET

"""
Cocke-Kasami-Younger (CKY) algorithm for bottom-up CFG parsing
Goals:
    > Write CKY algorithm and use it as a recognizer of CFG.
    > Extend it to a parser by adding back pointers
    > Get counts of all possible CKY parse trees for each sentence
    that is in the language of CFG
Functionalities:
    > Create CKY chart
    > Create CKY parsed trees
    > Get total runtime
"""

import nltk
import time
import argparse
import texttable
# import numpy as np
# import pandas as pd
# from nltk import CFG
from nltk.tree import Tree
from itertools import product
from collections import defaultdict
from nltk.grammar import is_nonterminal


def data_load(grammar_f, sents_f):
    """Load the CNF (Chomsky normalform) grammar and test sentence from the
    given files.

    Args:
        grammar_f (str): path to CNF grammar file
        sents_f (str): path to given test sentences

    Returns:
        object: nltk.grammar.CFG object containing the grammar.
        str: test sentences
    """
    grammar = nltk.data.load(grammar_f)
    sents = nltk.data.load(sents_f)

    return grammar, sents


def generate_production(grammar):
    """Convert CNF grammar into dictionary where keys are RHS of the
    rules/productions and values are it's (rules/productions) corresponding
    LHS.

    Args:
        grammar ([type]): Object of type "nltk.grammar.CFG " containing the CNF grammar

    Returns:
        dict: CNF grammar with all productions.
    """
    grammar_dict = {}
    for production in grammar.productions():
        rhs = production.rhs()
        if len(rhs) == 2 and is_nonterminal(rhs[0]) and is_nonterminal(rhs[1]):
            key = (rhs[0].symbol(), rhs[1].symbol())
            if key not in grammar_dict:
                grammar_dict[key] = []
            grammar_dict[key].append(production)

    return grammar_dict


def cky_parser(tokens, grammar):
    """This is the main CYK (Cocke–Younger–Kasami algorithm) algorithm. Given a
    sentence and CNF (Chomsky normal form) grammar, the goal is to determine if it belongs to the language of CFG.

    Args:
        tokens (list): list of tokens of the test sentence
        grammar (object): Object of type "nltk.grammar.CFG " containing the CNF grammar

    Returns:
        list: CYK parsed chart (upper triangular matrix) filled with non-terminal symbols.
        defaultdict: Dictionary of back pointers that will be used to reconstruct the parsed tree.
                    format: table[i][j]: {Node: [(x_index, y_index, left_child_node),(x_index, y_index, right_child_node)]}
    """

    grammar_dict = generate_production(grammar)
    n = len(tokens)

    chart = [[set() for col in range(n+1)] for row in range(n+1)]
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for j in range(1, n+1):  # Length of span
        # left to right: for each terms
        for rule in grammar.productions(rhs=tokens[j-1]):
            # for each rule in productions
            lhs_tag = rule.lhs()
            chart[n-j][j-1].add(lhs_tag)
            table[j-1][j][str(lhs_tag)].append(tokens[j-1])

        # go bottom-to-top
        for i in range(j-2, -1, -1):                    # Start of span
            for k in range(i+1, j):                     # Partition of span
                # get the right and down tags for each diagonal non-term symbol
                right_tags = chart[n-k][i]
                down_tags = chart[n-j][k]
                for rtag in right_tags:
                    for dtag in down_tags:
                        key = (str(rtag), str(dtag))
                        if key in grammar_dict:
                            for matched_rule in grammar_dict[key]:
                                chart[n-j][i].add(matched_rule.lhs())
                                # Backpointers: store childen nodes and locations (x,y indices)
                                l_child = (i, k, key[0])
                                r_child = (k, j, key[1])
                                children = (l_child, r_child)
                                table[i][j][str(matched_rule.lhs())].append(
                                    children)

    for line in chart:
        for idx, item in enumerate(line):
            if item == set():
                line[idx] = "."

    return chart, table


def cky_recognizer(chart, root, show_chart, show_tree):
    """Simple function that completes the CYK recognizer task. Given the CYK
    parsed chart, returns True if the given test sentence is in the language of CFG.

    Args:
        chart (list): CYK parsed chart (upper triangular matrix) filled with non-terminal symbols.
        root (object): Root or the start symbol of the given CNF grammar- "SIGMA"- type: nltk.grammar.Nonterminal
        show_chart (bool): Choice to display all CYK parsed charts. Default: False
        show_tree (bool): Choice to display all CYK parsed trees. Default: False

    Returns:
        bool: if the given test sentence lies in the language of CFG
    """
    if show_chart or show_tree:
        try:
            if root in chart[0][0]:
                print("Given sentence is in the language of CFG\n")
                return True
            else:
                print("Given sentence is not in the language of CFG\n")
                return False
        except TypeError:
            # special case when chart[0][0] contains only ".". Hence root-"SIGMA" missing in the root index
            print("Given sentence is not in the language of CFG\n")
            return False
    else:
        try:
            if root in chart[0][0]:
                return True
            else:
                return False
        except TypeError:
            return False


def print_chart(chart, tokens):
    """Given the CYK parsed chart and list of tokens, draw all possible CYK
    parsed charts in a formatted manner using Texttable.

    Args:
        chart (list): CYK parsed chart (upper triangular matrix) filled
                    with non-terminal symbols.
        tokens (list): list of tokens from the test sentence
    """
    nrow = len(tokens) + 1
    ncol = len(tokens)
    tokens = [""] + tokens
    t = texttable.Texttable()
    rows = [[" "] + list(map(str, range(1, ncol+1)))]

    for i in range(-1, nrow-1):
        row_content = [''.join(str(chart[i][j])) for j in range(ncol)]
        rows.append([str(len(tokens)-i)] + row_content)
    del rows[1]
    rows.insert(0, tokens)
    t.add_rows(rows)
    # length = [8] * len(rows[0])
    # t.set_cols_width(length)
    try:
        print(t.draw() + "\n")
    except ValueError:
        print("Couldn't draw the chart. Error: Too big/small to render")


def print_parsed_tree(table, tokens, root, show_tree):
    """Given the back pointer table, print all possible CYK parsed trees (in
    fancy ASCII art representation) using back pointers and return the total
    counts of parse trees.

    Args:
        table (dict): Dictionary of back pointers that will be used to reconstruct the parsed tree.
        tokens (list): List of tokens from the test sentence
        root (object): Root or the start sumbol of the given CNF grammar-
                        "SIGMA"- type: nltk.grammar.Nonterminal
        show_tree (bool): Choice to display all CYK parsed trees. Default: False
    Returns:
        int: Counts of all possible CYK parsed trees
    """
    my_trees = backtrace(table, 0, len(tokens), root)
    if show_tree:
        for tree in my_trees:
            tree.pretty_print(unicodelines=True, nodedist=4)
    tree_counts = len(list(my_trees))

    return tree_counts


def backtrace(table, i, j, root):
    """Given back pointer table, perform a recursive backtracing to trace back
    how a tree was constructed using the indices of the children nodes. In short
    the back pointer table has memory about all left and right child nodes of
    all non-terminal and terminal symbols in CYK chart.

    Args:
        table (dict): Dictionary of back pointers that will be used to reconstruct the parsed tree.
        i (int): column index of the root node
        j (int): length of tokens from the test sentence
        root (object): Root or the start symbol of the given CNF grammar- "SIGMA"- type: nltk.grammar.Nonterminal

    Returns:
        list: list of all possible CYK parsed tree
    """

    root = str(root)
    trees = []
    if len(table[i][j][root]) == 1 and isinstance(table[i][j][root][0], str):       # Terminal detected
        return [(Tree(root, [table[i][j][root][0]]))]
    else:
        for p1, p2 in table[i][j][root]:                # A-> BC. Get left(B) and right(C) child for a non-term symbol(A)
            i, k, B = p1
            ko, j, C = p2
            for left_tree, right_tree in product(backtrace(table, i, k, B), backtrace(table, ko, j, C)):    # Recursively back trace the left and right child for p1 and p2. 
                                                                                                            # Continue until a terminal symbol is found
                trees.append(Tree(root, [left_tree, right_tree]))

    # import pdb; pdb.set_trace()
    return trees

    ############## Small test example ##############
    # grammar = nltk.data.load('small_grammar.cfg')
    # sents = nltk.data.load('small_test_sent.txt')
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # small_grammar.cfg
    # S -> NP VP
    # VP -> VP PP
    # VP -> V NP
    # VP -> eats
    # PP -> P NP
    # NP -> DET N
    # NP -> "she"
    # V -> "eats"
    # P -> "with"
    # N -> "fish"
    # N -> "fork"
    # DET -> "a"
    # ***********************
    # small_test_sent.txt
    # she eats a fish with a fork
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def main():
    """Main function."""

    args = parse_arguments()
    start = time.time()
    print("Expected runtime: 20 s on ATIS dataset")
    grammar, sents = data_load(args.grammar_f, args.sents_f)
    test_sents = nltk.parse.util.extract_test_sentences(sents)

    # Display sentences and counts in a tabulated manner using texttable
    # References: https://pypi.python.org/pypi/texttable

    t = texttable.Texttable()
    length = [5, 70, 5, 10]
    t.set_cols_width(length)
    rows = [["S.No.", "test sentence", "CFG", "parse tree counts"]]

    cyk_runtime = 0
    bp_runtime = 0

    for index, sent in enumerate(test_sents):
        tokens = sent[0]
        sentence = " ".join(tokens)
        if args.show_chart or args.show_tree:
            print("(", index, ")", " ".join(tokens), "\n")
        root = grammar.start()

        cyk_start = time.time()
        chart, table = cky_parser(tokens, grammar)
        cyk_end = time.time()
        cyk_runtime += cyk_end - cyk_start

        tree_counts = 0
        next_row = [str(index+1), str(sentence),
                    str("False"), str(tree_counts)]

        if cky_recognizer(chart, root, args.show_chart, args.show_tree):
            if args.show_chart:
                print_chart(chart, tokens)

            bp_start = time.time()
            tree_counts = print_parsed_tree(
                table, tokens, root, args.show_tree)
            bp_end = time.time()
            bp_runtime = bp_end - bp_start

            next_row = [str(index+1), str(sentence),
                        str("True"), str(tree_counts)]

        if args.show_chart or args.show_tree:
            dash = "-="*40
            print(dash, '\n')

        rows.append(next_row)
    if args.show_summary:
        t.add_rows(rows)
        print(t.draw())

    end = time.time()

    print("Total runtime: %.3f s" % (end-start))
    print("CYK parser runtime: %.3f s" % cyk_runtime)
    print("Backpointer runtime: %.6f s" % bp_runtime)


def parse_arguments():
    """parse arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grammar_f", help="path to grammar file")
    parser.add_argument("sents_f", help="path test sentences file")
    parser.add_argument("-show_chart", default=False,
                        type=bool, help='display CYK parsed chart')
    parser.add_argument("-show_tree", default=False,
                        type=bool, help='display CYK parsed tree')
    parser.add_argument("-show_summary", default=True,
                        type=bool, help='display summary table of tree counts')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

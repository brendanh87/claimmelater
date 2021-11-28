import gzip
import numpy as np
import json as js
from sklearn import svm
from sklearn import datasets

# Builds a dictionary with three fields id pair and same, which each contain a list where each list index is a docpair
def get_data(inputs_file_path, labels_file_path):
    
    # create entries that are lists of ids, boolean flags, and docpairs
    di = {}
    di['id'] = []
    di['same'] = []
    di['pair'] = []

    # get the number of examples
    count = 0

    # read the inputs and labels
    with open(inputs_file_path) as f, open(labels_file_path) as g:
        # loop through the inputs to get the id and the pair of text
        for line in f:
            # increment count
            count += 1
            data = js.loads(line)
            di['id'].append(data['id'])
            # TODO: instead of adding on the whole pair, add on ngrams
            di['pair'].append(data['pair'])
        # loop through the labels to get the boolean flag (whether or not they come from the same author)
        for line in g:
            data = js.loads(line)
            di['same'].append(data['same'])

    # return the dictionary and the number of examples
    return di, count
  
get_data("data/pan20-authorship-verification-training-small.jsonl", "data/pan20-authorship-verification-training-small-truth.jsonl")    
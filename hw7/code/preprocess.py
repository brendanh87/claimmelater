import gzip
import numpy as np
import json as js

# Builds a dictionary with three fields id pair and same, which each contain a list where each list index is a docpair
def get_data(inputs_file_path, labels_file_path):
    di = {}
    di['id'] = []
    di['same'] = []
    di['pair'] = []
    with open(inputs_file_path) as f, open(labels_file_path) as g:
        for line in f:
            data = js.loads(line)
            di['id'].append(data['id'])
            di['pair'].append(data['pair'])

        for line in g:
            data = js.loads(line)
            di['same'].append(data['same'])

    print(di['id'][0])
    print(di['same'][0])
    print(di['pair'][0])
  
get_data("data/pan20-authorship-verification-training-small.jsonl", "data/pan20-authorship-verification-training-small-truth.jsonl")    
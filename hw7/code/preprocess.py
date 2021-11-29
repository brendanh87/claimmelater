import gzip
import numpy as np
import json as js
from sklearn.feature_extraction.text import CountVectorizer

# Builds a dictionary with three fields id pair and same, which each contain a list where each list index is a docpair
def get_data(inputs_file_path, labels_file_path):

    # create entries that are lists of ids and boolean flags
    di = {}
    di['id'] = []
    di['same'] = []

    # get the number of examples
    count = 0

    # initialize the vectorizer
    vectorizer = CountVectorizer(input='content', ngram_range=(1,3), analyzer='char', min_df=0.01)

    # initialize an array to hold all the separate texts
    arr = []

    # read the inputs and labels
    with open(inputs_file_path) as f, open(labels_file_path) as g:
        # loop through the inputs to get the id and the pair of text
        for line in f:
            print(count)
            # increment count
            count += 1
            data = js.loads(line)
            di['id'].append(data['id'])
            # add the first 10000 characters of each text to the array
            arr.append(data['pair'][0][0 : 10000])
            arr.append(data['pair'][1][0 : 10000])

        # loop through the labels to get the boolean flag (whether or not they come from the same author)
        for line in g:
            data = js.loads(line)
            di['same'].append(data['same'])

    # get the features!
    features = vectorizer.fit_transform(arr).toarray()     

    # load into txt file
    with open("hw7/code/count-vectors.txt", 'w') as f:
        f.truncate()
        np.savetxt(f, features)  

    # return the dictionary and the number of examples
    return di, count, features

# to save time, this function reads the count-vectors.txt file to get the features
def read_count_vectors(vectors_file_path):
    features = np.loadtxt(vectors_file_path)
    print(features.shape)

read_count_vectors("hw7/code/count-vectors.txt")
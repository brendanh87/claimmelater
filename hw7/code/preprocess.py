import gzip
import numpy as np
import json as js
from sklearn.feature_extraction.text import CountVectorizer

# Builds a dictionary with three fields id pair and same, which each contain a list where each list index is a docpair
def get_data(inputs_file_path, labels_file_path):

    # get the number of examples
    count = 0

    # initialize the vectorizer
    vectorizer = CountVectorizer(input='content', ngram_range=(1,3), analyzer='char', min_df=0.01)

    # initialize arrays to hold all the separate texts, labels
    arr = []
    labels = []

    # read the inputs and labels
    with open(inputs_file_path) as f, open(labels_file_path) as g:
        # loop through the inputs to get the id and the pair of text
        for line in f:
            print(count)
            # increment count
            count += 1
            data = js.loads(line)
            # add the first 10000 characters of each text to the array
            arr.append(data['pair'][0][0 : 10000])
            arr.append(data['pair'][1][0 : 10000])

        # loop through the labels to get the boolean flag (whether or not they come from the same author)
        for line in g:
            data = js.loads(line)
            labels.append(data['same'])

    # get the features!
    features = vectorizer.fit_transform(arr).toarray()     

    # load features into a .npy file
    with open('hw7/code/count-vectors.npy', 'wb') as f:
        f.truncate()
        np.save(f, features)

    # load labels into a .npy file
    with open('hw7/code/labels.npy', 'wb') as f:
        f.truncate()
        np.save(f, labels)     

    # return the dictionary, num_examples, and the features
    return count, features

# to save time, this function reads the count-vectors.txt file to get the features
def read_data(vectors_file_path, labels_file_path):
    # get the features and labels from the .npy file
    features = np.load(vectors_file_path)
    labels = np.load(labels_file_path)

    # trim the data
    features_to_cut_off = len(features) % 10
    labels_to_cut_off = features_to_cut_off // 2

    features = features[:-features_to_cut_off]
    labels = labels[:-labels_to_cut_off]

    # split the training/testing data 70/30
    train_inputs, test_inputs = np.split(features, [int(0.7 * len(features))])
    train_labels, test_labels = np.split(labels, [int(0.7 * len(labels))])

    # reshape the inputs into pairs
    train_inputs = np.reshape(train_inputs, (-1, 2, train_inputs.shape[-1]))
    test_inputs = np.reshape(test_inputs, (-1, 2, test_inputs.shape[-1]))

    return train_inputs, test_inputs, train_labels, test_labels


# TODO: run get_data() AFTER creating the files count-vectors.npy and labels.npy to load it on your device
get_data('/Users/alyssamarie/Desktop/School/cs1470/claimmelater/data/pan20-authorship-verification-training-small.jsonl', '/Users/alyssamarie/Desktop/School/cs1470/claimmelater/data/pan20-authorship-verification-training-small-truth.jsonl') 
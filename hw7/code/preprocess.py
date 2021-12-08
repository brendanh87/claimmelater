import gzip
import numpy as np
import json as js
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf

# Builds a dictionary with three fields id pair and same, which each contain a list where each list index is a docpair
def get_data(inputs_file_path, labels_file_path):
    """
    Takes the inputs and labels and returns an array with each entry's count vectorization.

    :inputs_file_path: the file path of the inputs
    :labels_file_path: the file path of the labels
    """

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
    with open('count-vectors.npy', 'wb') as f:
        f.truncate()
        np.save(f, features)

    # load labels into a .npy file
    with open('labels.npy', 'wb') as f:
        f.truncate()
        np.save(f, labels)     

    # return the dictionary, num_examples, and the features
    return count, features

def read_data(vectors_file_path, labels_file_path):
    """
    Reads the features and labels from given files.

    :vectors_file_path: the file path to the vectors
    :labels_file_path: the file path to the labels
    """

    # get the features and labels from the .npy file
    features = np.load(vectors_file_path)
    labels = np.load(labels_file_path)

    # trim the data
    features_to_cut_off = len(features) % 10
    labels_to_cut_off = features_to_cut_off // 2

    features = features[:-features_to_cut_off]
    labels = labels[:-labels_to_cut_off]

    # reshape the features into pairs
    features = np.reshape(features, (-1, 2, features.shape[-1]))

    # shuffle the data
    randomize = tf.random.shuffle(tf.range(len(features)))
    features = tf.gather(features, randomize)
    labels = tf.gather(labels, randomize)

    # split the training/testing data 70/30
    train_inputs, test_inputs = np.split(features, [int(0.7 * len(features))])
    train_labels, test_labels = np.split(labels, [int(0.7 * len(labels))])

    return train_inputs, test_inputs, train_labels, test_labels

# USER TODO: run get_data() AFTER creating the files count-vectors.npy and labels.npy to load it on your device

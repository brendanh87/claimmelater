import numpy as np
import os
import random
import tensorflow as tf
from preprocess import get_data
from preprocess import read_data
from model import Model

def main():
    # intialize the model
    model = Model()
    # get the inputs and labels for training and testing
    train_inputs, test_inputs, train_labels, test_labels = read_data('hw7/code/count-vectors.npy', 'hw7/code/labels.npy')

    # train the model with stochastic gradient descent
    for batch in range(model.batch_size):
        # store starting and ending indices for batch
        start = batch_number * model.batch_size
        end = start + model.batch_size
        # gradient descend that bad boi
        model.call(train_inputs[start : end], train_labels[start : end])

    # test the accuracy of the model

import numpy as np
import os
import random
import tensorflow as tf
from preprocess import get_data
from preprocess import read_count_vectors
from model import Model

def train(model, train_inputs, train_labels):
    # stochastic gradient descent
    for batch in range(model.batch_size):
        # store starting and ending indices for batch
        start = batch_number * model.batch_size
        end = start + model.batch_size
        # gradient descend that bad boi
        model.call(train_inputs[start : end], train_labels[start : end])


def test(model, test_inputs, test_labels):
    pass

def main():
    # intialize the model
    model = Model()
    # get the train_inputs, train_labels
    train_inputs, train_labels = get_data('data/pan20-authorship-verification-training-small.jsonl', 'data/pan20-authorship-verification-training-small-truth.jsonl')
    
    # stochastic gradient descent
    for batch in range(model.batch_size):
        # store starting and ending indices for batch
        start = batch_number * model.batch_size
        end = start + model.batch_size
        # gradient descend that bad boi
        model.call(train_inputs[start : end], train_labels[start : end])

    
    test(model, test_inputs, test_labels)
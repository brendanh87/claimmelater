from preprocess import vectorize
from model import Model, load_classifier_model
import string
import numpy as np
import tensorflow as tf
import re

DICT_FILE_PATH = "output_dictionary.pkl"
PATH_TO_WEIGHTS = "whole_model_weights"
PATH_TO_FILE_A = "alice.txt"
PATH_TO_FILE_B = "deathlyhallows.txt"

# def lower_n_unpunc(text):
#     return text.translate(str.maketrans('','', string.punctuation)).replace('\n', ' ').replace("  "," ").lower()

def predict_for_texts(path_to_file_a, path_to_file_b):
    with open(path_to_file_a, 'r') as file_a:
       text_a = re.sub('\s+',' ', file_a.read().replace('\n', ' ')[:10000])
    
    with open(path_to_file_b, 'r') as file_b:
       text_b = re.sub('\s+',' ', file_b.read().replace('\n', ' ')[:10000])

    vectors = np.array(vectorize(DICT_FILE_PATH, [text_a, text_b]))
    model = Model()
    load_classifier_model(model, PATH_TO_WEIGHTS)
    print(model.call(tf.expand_dims(vectors[0], 0), tf.expand_dims(vectors[1], 0)))

predict_for_texts(PATH_TO_FILE_A, PATH_TO_FILE_B)


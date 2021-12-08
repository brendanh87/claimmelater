import sys
import os
import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import metrics
import matplotlib.pyplot as plt
from preprocess import read_data
from tensorflow.keras.utils import plot_model

class Model(tf.keras.Model):
 
    def __init__(self, is_convolution=False):
        """
        The Model class predicts if the two texts are written by the same author.

        Our siamese network is made up for two identical subnetworks with the first
        and second example are sent into. These subnetworks contain a number of 
        residual blocks: 
        - If is_convolution is set to False, these blocks are made
        up of dense layers. 
        - If is_convolution is set to True, these blocks are made
        up of convolutional layers.

        We then take the absolute distance between the outputs of the subnetworks.

        On top of the siamese network, we add a classifier network, which passes the 
        output of the siamese network into a few dense layers and then into a binary
        classifier.
        """

        super(Model, self).__init__()

        # intialize hyperparameters and optimizer
        self.learning_rate = 0.001
        self.epochs = 20
        self.siamese_epochs = 80
        self.res_layer_count = 8
        self.embedding_size = (8736)
        self.batch_size = 120
        self.is_convolution = is_convolution

        self.optimizer = tfa.optimizers.RectifiedAdam(self.learning_rate)

        # subnetwork model
        subnetwork_input = keras.Input(shape = self.embedding_size)
        x = layers.BatchNormalization()(subnetwork_input)
        x = layers.GaussianNoise(stddev=0.00001)(x)
        x = layers.Dropout(rate=0.9)(x)
        x = layers.Dense(units=512)(x)

        # residual blocks
        if is_convolution == True:
            x = tf.expand_dims(x, axis=2)
            for i in range(self.res_layer_count):
                x = self.residual_block_conv(x)
        else:
            for i in range(self.res_layer_count):
                x = self.residual_block(x)             

        # set up the subnetwork
        self.subnetwork = keras.Model(inputs = subnetwork_input, outputs = x, name = "subnetwork")

        # siamese network
        siameseA_input = keras.Input(shape = self.embedding_size, name = "encoded_doc_A")
        siameseA_output = self.subnetwork(siameseA_input)
        siameseB_input = keras.Input(shape = self.embedding_size, name = "encoded_doc_B")
        siameseB_output = self.subnetwork(siameseB_input)
        subtracted = layers.Subtract()([siameseA_output, siameseB_output])
        distance = Lambda(tf.norm, output_shape = (1), name = "euclid_distance")(subtracted)

        # set up the siamese network
        self.siamese = keras.Model(inputs = [siameseA_input, siameseB_input], 
                                   outputs = distance,
                                   name = 'siamese')
        # classifier model
        z = layers.Subtract()([siameseA_output, siameseB_output])
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        classifier_output = layers.Dense(1, activation = 'sigmoid')(z)

        # set up the cumulative classifier model
        self.classifier = keras.Model(inputs = [siameseA_input, siameseB_input], outputs = classifier_output, name = 'classifier')

    def call(self, inputs):
        """
        Calls the model on a set of inputs
        :param inputs: A list of two embedding_size vectors
        :return: A [0,1] float value representing confidence of whether the two inputs are from the same author
        """
        return self.classifier([inputs[0], inputs[1]])

    def residual_block(self, input):
        """
        This residual block is a series of dense layers.
        """
        x = layers.Dense(512)(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Subtract()([x, input])
        x = layers.Activation('relu')(x)
        return x  

    def residual_block_conv(self, input):
        """
        This residual block is made up of Conv1D layer(s).
        """
        x = layers.Conv1D(32, 3, padding = 'same', input_shape=(512, 1))(input)
        x = layers.BatchNormalization()(x)
        x = layers.Subtract()([x, input])
        x = layers.Activation('relu')(x)
        return x       

def f1(y_true, y_pred):
    """
    Calculates the f1 score based on the predictions and the labels

    """

    def recall(y_true, y_pred):
        """
        Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    # calculate the f1 score based on precision and recall
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision*recall)/(precision+recall+K.epsilon()))

def train_whole_model(model, train_inputs, train_labels, freeze_siamese=False):
    """
    Train the entire model for the specified number of epochs.

    :param model: the model
    :param train_inputs: training inputs in the shape (batch_size, 2, 8736)
    :param train_labels: training labels in the shape (batch_size)
    :param freeze_siamese: if False, train the whole model, if True, only train the classifier network
    :return history: history of the model for visualization
    """
    # compile the model
    model.classifier.compile(loss = keras.losses.BinaryCrossentropy(), optimizer = model.optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    if freeze_siamese == True:
        model.siamese.trainable = False
    # train the model
    history = model.classifier.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, epochs = model.epochs, batch_size = model.batch_size)
    model.classifier.save_weights('whole_model_weights')
    # return history
    return history

def train_siamese_network(model, train_inputs, train_labels):
    """
    Train the siamese network for the specified number of epochs.

    :param model: the model
    :param train_inputs: training inputs in the shape (batch_size, 2, 8736)
    :param train_labels: training labels in the shape (batch_size)
    :return siamese_history: history of the siamese_network for visualization
    """
    # # compile the siamese model
    model.siamese.compile(loss = tfa.losses.contrastive_loss, optimizer = model.optimizer)
    # train the siamese model
    siamese_history = model.siamese.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, epochs=model.siamese_epochs, batch_size = model.batch_size)
    model.siamese.save_weights('siamese_model_weights')
    # return siamese history
    return siamese_history

def load_classifier_model(model, path_to_weights):
    model.classifier.compile(loss = keras.losses.BinaryCrossentropy(), optimizer = model.optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    model.classifier.load_weights(path_to_weights).expect_partial()    

def test(model, test_inputs, test_labels):
    """
    Evaluates the entire model.

    :param model: the model
    :param test_inputs: testing inputs in the shape (batch_size, 2, 8736)
    :param test_labels: testing labels in the shape (batch_size)
    """
    return model.classifier.evaluate([test_inputs[:, 0], test_inputs[:, 1]], test_labels, verbose=2)

def visualize_data(history):
    """
    Plots the loss over epochs in training.

    :param history: the model history
    """   
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 1.2)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def main():
    # get the data from the specified files
    train_inputs, test_inputs, train_labels, test_labels = read_data('count-vectors.npy', 'labels.npy')   

    if sys.argv[1] not in {"CONV", "DENSE"}:
        print("USAGE: python model.py <Model Type> <Training Type> <Train/Load>")
        print("<Model Type>: [CONV/DENSE]")
        exit()

    if sys.argv[2] not in {"WHOLE", "SPLIT", "FREEZESPLIT"}:
        print("USAGE: python model.py <Model Type> <Training Type> <Train/Load>")
        print("<Training Type>: [WHOLE/SPLIT/FREEZESPLIT]")
        exit()

    if sys.argv[3] not in {"TRAIN", "LOAD"}:     
        print("USAGE: python model.py <Model Type> <Training Type> <Train/Load>")
        print("<Train/Load>: [TRAIN/LOAD]")
        exit()

    # create model based on parameter
    if sys.argv[1] == "CONV":
        model = Model(is_convolution=True)
    elif sys.argv[1] == "DENSE":
        model = Model()  

    # load weights
    if sys.argv[3] == "LOAD":
        whole_model_path = input("Path to weights (dir/weight_names): ")
        load_classifier_model(model, whole_model_path)

    else:
        # train based on input
        if sys.argv[2] == "WHOLE":
            history = train_whole_model(model, train_inputs, train_labels)
        elif sys.argv[2] == "SPLIT":
            train_siamese_network(model, train_inputs, train_labels)    
            history = train_whole_model(model, train_inputs, train_labels)
        elif sys.argv[2] == "FREEZESPLIT":
            train_siamese_network(model, train_inputs, train_labels)   
            history = train_whole_model(model, train_inputs, train_labels, freeze_siamese=True)
        
        # graph loss over training
        visualize_data(history)   

    # test the model
    test(model, test_inputs, test_labels)
  
if __name__ == '__main__':
    main()
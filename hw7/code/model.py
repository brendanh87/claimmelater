import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from preprocess import get_data


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts if the two texts are written by the same author.
        """

        super(Model, self).__init__()

        # intialize the optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.siamese_hidden_dim = 512
        self.classifier_hidden_dim = 256

        # initialize the siamese network
        self.siamese = f.keras.Sequential()
        self.siamese.add(BatchNormalization())
        self.siamese.add(GaussianNoise(stddev=0.1))
        self.siamese.add(Dropout(rate=0.3))

        # TODO: add the residual network to siamese model
        self.siamese.add(tf.keras.layers.Dense(self.siamese_hidden_dim))

        # initialize the classifier
        self.classifier = tf.keras.Sequential()
        self.classifier.add(tf.keras.layers.Dense(self.classifier_hidden_dim, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    def residual_block(self, inputs):
        pass    
        

    def call(self, inputs):
        pass


    def loss(self, feature1, feature2, labels, margin=1.0):
        # get the difference of the two feature vectors
        predictions = tf.linalg.norm(feature1 - feature2, axis=1)
        # get the contrastive loss
        loss = tf.math.reduce_mean(tfa.losses.contrastive_loss(labels, predictions, margin=margin))
        return loss

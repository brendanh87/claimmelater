import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import Lambda
from preprocess import get_data
from tensorflow.keras import metrics


class Model(tf.keras.Model):
 
    def __init__(self, vocab_size):
        """
        The Model class predicts if the two texts are written by the same author.
        """

        super(Model, self).__init__()

        # intialize the optimizer
        self.learning_rate = 0.001
        self.res_layer_count = 8
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.embedding_size = (1, 11675)
        self.batch_size = 120
        # self.siamese_hidden_dim = 512
        # self.classifier_hidden_dim = 256

        # Subnetwork model
        subnetwork_input = keras.Input(self.embedding_size)
        x = layers.BatchNormalization()(subnetwork_input)
        x = layers.GaussianNoise(stddev=0.1)(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1,512)(x)

        # residual layer
        for i in range(self.res_layer_count):
            x = self.residual_block(x)

        self.subnetwork = keras.Model(inputs = subnetwork_input, 
                                      outputs = x, name = 'subnetwork')

        # Siamese network
        siameseA_input = keras.Input(self.embedding_size)
        siameseA_output = self.subnetwork(siameseA_input)
        siameseB_input = keras.Input(self.embedding_size)
        siameseB_output = self.subnetwork(siameseB_input)
        distance = Lambda(lambda x, y: tf.linalg.norm(x - y, axis = 1))([siameseA_output, siameseB_output])
        self.siamese = keras.Model(inputs = [siameseA_input, siameseB_input], 
                                   outputs = distance,
                                   name = 'siamese')
        # classifier model
        siameseA_output = keras.Input(shape = (1, 512))
        siameseB_output = keras.Input(shape = (1, 512))
        z = layers.Subtract([subnetwork_one_input, subnetwork_two_input])
        z = layers.Dense((1,512), activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense((1,512), activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense((1,512), activation = 'relu')(z)
        classifier_output = layers.Dense(1, activation = 'sigmoid')
        self.classifier = keras.Model(inputs = [siameseA_output, siameseB_output], output = classifier_output, name = 'classifier')

    def residual_block(self, input):
        x = layers.Dense((1,512))(input)
        x = layers.BatchNormalization(x)
        x = layers.Dense((1,512))(x)
        x = layers.BatchNormalization(x)
        x = layers.Subtract()([x, input])
        x = layers.Activation('relu')(x)

        return x   

    def call(self, inputs, labels):
        self.siamese.compile(loss = metrics.contrastive_loss, optimizer = self.optimizer)
        self.siamese.fit([inputs[0], inputs[1]], labels, batch_size = self.batch_size)


    def loss(self, feature1, feature2, labels, margin=1.0):
        # get the difference of the two feature vectors
        # get the contrastive loss
        loss = tf.math.reduce_mean(tfa.losses.contrastive_loss(labels, predictions, margin=margin))
        return loss
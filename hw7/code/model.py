import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda
from tensorflow.keras import metrics
from preprocess import read_data

class Model(tf.keras.Model):
 
    def __init__(self):
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

        # Subnetwork model
        subnetwork_input = keras.Input(shape = (1, 11675))
        x = layers.BatchNormalization()(subnetwork_input)
        x = layers.GaussianNoise(stddev=0.1)(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(units = 512)(x)

        # residual layer
        for i in range(self.res_layer_count):
            x = self.residual_block(x)

        self.subnetwork = keras.Model(inputs = subnetwork_input, outputs = x, name = "subnetwork")

        # Siamese network
        siameseA_input = keras.Input(shape = (1, 11675), name = "encoded_doc_A")
        siameseA_output = self.subnetwork(siameseA_input)
        siameseB_input = keras.Input(shape = (1, 11675), name = "encoded_doc_B")
        siameseB_output = self.subnetwork(siameseB_input)
        subtracted = layers.Subtract()([siameseA_output, siameseB_output])
        distance = Lambda(tf.norm, output_shape = (1), name = "euclid_distance")(subtracted)
        self.siamese = keras.Model(inputs = [siameseA_input, siameseB_input], 
                                   outputs = distance,
                                   name = 'siamese')
        # classifier model
        siameseA_output = keras.Input(shape = (1, 512))
        siameseB_output = keras.Input(shape = (1, 512))
        z = layers.Subtract()([siameseA_output, siameseB_output])
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        classifier_output = layers.Dense(1, activation = 'sigmoid')(z)
        self.classifier = keras.Model(inputs = [siameseA_output, siameseB_output], outputs = classifier_output, name = 'classifier')

    def residual_block(self, input):
        x = layers.Dense(512)(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Subtract()([x, input])
        x = layers.Activation('relu')(x)

        return x   

    # def call(self, inputs, labels):
    #     self.siamese.compile(loss = metrics.contrastive_loss, optimizer = self.optimizer)
    #     self.siamese.fit([inputs[0], inputs[1]], labels, batch_size = self.batch_size)


    # def loss(self, feature1, feature2, labels, margin=1.0):
    #     # get the difference of the two feature vectors
    #     # get the contrastive loss
    #     loss = tf.math.reduce_mean(tfa.losses.contrastive_loss(labels, predictions, margin=margin))
    #     return loss

def main():
    model = Model()
    
    train_inputs, test_inputs, train_labels, test_labels = read_data('hw7/code/count-vectors.npy', 'hw7/code/labels.npy')

    model.siamese.compile(loss = tfa.losses.contrastive_loss, optimizer = model.optimizer)
    model.siamese.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, batch_size = model.batch_size)

if __name__ == '__main__':
    main()
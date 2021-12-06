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

class Model(tf.keras.Model):
 
    def __init__(self):
        """
        The Model class predicts if the two texts are written by the same author.
        """

        super(Model, self).__init__()

        # intialize the optimizer
        self.learning_rate = 0.01
        self.res_layer_count = 8
        self.optimizer = tfa.optimizers.RectifiedAdam(self.learning_rate)
        self.embedding_size = (8736)
        self.batch_size = 120

        # Subnetwork model
        subnetwork_input = keras.Input(shape = self.embedding_size)
        x = layers.BatchNormalization()(subnetwork_input)
        x = layers.GaussianNoise(stddev=0.00001)(x)
        x = layers.Dropout(rate=0.9)(x)
        x = layers.Dense(units = 512)(x)

        # residual layer
        for i in range(self.res_layer_count):
            x = self.residual_block(x)

        self.subnetwork = keras.Model(inputs = subnetwork_input, outputs = x, name = "subnetwork")

        # Siamese network
        siameseA_input = keras.Input(shape = self.embedding_size, name = "encoded_doc_A")
        siameseA_output = self.subnetwork(siameseA_input)
        siameseB_input = keras.Input(shape = self.embedding_size, name = "encoded_doc_B")
        siameseB_output = self.subnetwork(siameseB_input)
        subtracted = layers.Subtract()([siameseA_output, siameseB_output])
        distance = Lambda(tf.norm, output_shape = (1), name = "euclid_distance")(subtracted)
        self.siamese = keras.Model(inputs = [siameseA_input, siameseB_input], 
                                   outputs = distance,
                                   name = 'siamese')
        # classifier model
        # siameseA_output = keras.Input(shape = (1, 512))
        # siameseB_output = keras.Input(shape = (1, 512))
        z = layers.Subtract()([siameseA_output, siameseB_output])
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        z = layers.BatchNormalization()(z)
        z = layers.Dense(512, activation = 'relu')(z)
        classifier_output = layers.Dense(1, activation = 'sigmoid')(z)
        # self.classifier = keras.Model(inputs = [siameseA_output, siameseB_output], outputs = classifier_output, name = 'classifier')
        self.classifier = keras.Model(inputs = [siameseA_input, siameseB_input], outputs = classifier_output, name = 'classifier')

    def residual_block(self, input):
        x = layers.Dense(512)(input)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(512)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Subtract()([x, input])
        x = layers.Activation('relu')(x)

        return x   

def f1(y_true, y_pred):

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def main():
    # instantiate model
    model = Model()
    
    # get the data from preprocessing
    train_inputs, test_inputs, train_labels, test_labels = read_data('count-vectors.npy', 'labels.npy')

    # compile the model
    model.classifier.compile(loss = keras.losses.BinaryCrossentropy(), optimizer = model.optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    # train the model
    history = model.classifier.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, epochs = 5, batch_size = model.batch_size)
    # test the model
    test_scores = model.classifier.evaluate([test_inputs[:, 0], test_inputs[:, 1]], test_labels, verbose=2)

    # ====== METHOD 2: TRAINING THE SIAMESE SEPARATELY =====

    # ---- TRAINING AND SAVING: COMMENT OUT IF LOADING IN SIAMESE ----
    # compile the siamese model
    model.siamese.compile(loss = tfa.losses.contrastive_loss, optimizer = model.optimizer)
    # train the siamese model
    siamese_history = model.siamese.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, epochs=model.epochs, batch_size = model.batch_size)
    model.siamese.save_weights('siamese_model_weights')

    # summarize history for loss
    plt.plot(siamese_history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 1.2)
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #---------------

    # # ---- LOADING SIAMESE: COMMENT OUT IF TRAINING AND SAVING ----
    # model.siamese.load_weights('siamese_model_weights')
    # # ----------------------

    # ===== METHOD 2B: FREEZING SIAMESE NETWORK BEFORE TRAINING CLASSIFIER =====
    model.siamese.trainable = False
    # ===================================

    # ---- TRAINING AND SAVING CLASSIFIER: COMMENT OUT IF LOADING IN ----
    # compile classifier model
    model.classifier.compile(loss = keras.losses.BinaryCrossentropy(), optimizer = model.optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    all_history = model.classifier.fit([train_inputs[:, 0], train_inputs[:, 1]], train_labels, epochs=model.siamese_epochs, batch_size = model.batch_size)
    model.classifier.save_weights('split_model_weights')

    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.ylim(0, 1.2)
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

  
if __name__ == '__main__':
    main()
import Keras_ResNet50
import Keras_Inception_v3
import Keras_Inception_Resnet_v2
import model_utils_bbbc014 as mu
from keras.optimizers import Adam, SGD
import numpy as np
import pandas as pd
import csv
from PIL import Image
import cv2
#from skimage.transform import resize

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras import layers
from keras import regularizers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import load_model


class CNN_Model(object):

    def __init__(self,
                 cnn_model = "ResNet50",
                 dims  = (256,256,3),
                 regularization=0.0,
                 epochs = 1,
                 batch_size = 8,
                 lr=0.001,
                 momentum = 0.9,
                 weights = "imagenet",
                 images_path = "./"):

        self.cnn_model = cnn_model
        self.dims  = dims
        self.regularization=regularization
        self.epochs =  epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weights = weights
        self.model = None
        self.path = images_path

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def extend_model(self):

        """
        Classification block.

        """

        if self.cnn_model == "ResNet50":
            model = Keras_ResNet50.ResNet50(input_shape=self.dims, regularizer=self.regularization, weights=self.weights)
        elif self.cnn_model == "Inception_v3":
            model = Keras_Inception_v3.InceptionV3(input_shape=self.dims, regularizer=self.regularization, weights=self.weights)
        elif self.cnn_model == "Inception_Resnet_v2":
            model = Keras_Inception_Resnet_v2.InceptionResNetV2(input_shape=self.dims, regularizer=self.regularization, weights=self.weights)
        else:
            raise ValueError("cnn_model argument should be either 'ResNet50', 'Inception_v3' or 'Inception_Resnet_v2'")

        x = GlobalAveragePooling2D(name='avg_pool')(model.output)
        x = Dense(1, kernel_regularizer=regularizers.l2(self.regularization), activation='sigmoid', name='predictions')(x)
        extended_model = Model(inputs=model.input, outputs=x)

        self.model = extended_model
        return extended_model

    def fit_model(self):
        self.extend_model()

        # load labels; 512 labels, first 256 instances are MCF7 cell lines, second 256 instances are A549 cell lines.
        #labels      = np.load("bbbc014_labels.npy")
        sample_all  = np.array(range(1024))
        sample_MCF7 = np.array(range(0, 512))
        sample_A549 = np.array(range(512, 1024))

        #height, width, channels = self.dims

        # Two training sessions, one with MCF7 cell line, and one with A549 cell line.
        for sample in sample_A549, sample_MCF7:
            index_test, index_train = sample, np.array([x for x in sample_all if x not in sample])
            # Compile model, with SGD optimizer
            sgd = SGD(lr=self.lr, momentum=self.momentum)
            self.model.compile(optimizer=sgd, loss="binary_crossentropy", metrics = ["accuracy"])

            steps = index_train.shape[0]/self.batch_size
            self.model.fit_generator(mu.generator(index_train, self.batch_size, steps, dims=self.dims),
                                steps_per_epoch=steps, epochs=self.epochs, verbose=1, max_queue_size=4)

        return self.model

    def eval_model(self):
        # load labels; 512 labels, first 256 instances are MCF7 cell lines, second 256 instances are A549 cell lines.
        labels      = np.load("bbbc014_labels.npy")
        #sample_all  = np.array(range(1024))
        sample_MCF7 = np.array(range(0, 512))
        sample_A549 = np.array(range(512, 1024))

        height, width, channels = self.dims

        # Two training sessions, one with MCF7 cell line, and one with A549 cell line.
        for sample in sample_A549, sample_MCF7:

            # Create test set; ith position in labels correspond to ith image.
            X_test = np.zeros((sample.shape[0], height, width, channels))
            Y_test = labels[sample]

            for j,i in enumerate(sample):
                image = Image.open(self.path + "%s.png" % str(i))
                image = np.array(image)
                X_test[j] = image

            # Predict test set and obtain probabilities
            probs = self.model.predict(X_test, batch_size=self.batch_size, verbose=1)
            preds = np.zeros((32,))
            trues = np.zeros((32,))
            probas = np.zeros((32,))
            for j,i in enumerate(range(0, 512, 16)):
                preds[j] = 1 if np.mean(probs[i:i+16]) >= 0.5 else 0
                trues[j]  = Y_test[i]
                probas[j] = np.mean(probs[i:i+16])

            print('confusion matrix & classification report for test data')
            print(confusion_matrix(trues, preds))
            print(classification_report(trues, preds, output_dict=True))

        return

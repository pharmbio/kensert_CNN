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


class CNN_Model(object):

    def __init__(self,
                 cnn_model = "ResNet50",
                 dims  = (256,256,3),
                 regularization=0.0,
                 epochs = 1,
                 batch_size = 8,
                 lr=0.001,
                 momentum = 0.9,
                 weights = "imagenet"):

        self.cnn_model = cnn_model
        self.dims  = dims
        self.regularization=regularization
        self.epochs =  epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weights = weights


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

        return extended_model

    def fit_and_eval(self):
        # load labels; 512 labels, first 256 instances are MCF7 cell lines, second 256 instances are A549 cell lines.
        labels      = np.load("bbbc014_labels.npy")
        sample_all  = np.array(range(1024))
        sample_MCF7 = np.array(range(0, 512))
        sample_A549 = np.array(range(512, 1024))

        height, width, channels = self.dims

        # Two training sessions, one with MCF7 cell line, and one with A549 cell line.
        for sample in sample_A549, sample_MCF7:
            index_test, index_train = sample, np.array([x for x in sample_all if x not in sample])

            model = self.extend_model()
            # Compile model, with SGD optimizer
            sgd = SGD(lr=self.lr, momentum=self.momentum)
            model.compile(optimizer=sgd, loss="binary_crossentropy", metrics = ["accuracy"])

            path = "images_bbbc014/bbbc014_"
            # Create test set; ith position in labels correspond to ith image.
            X_test = np.zeros((sample.shape[0], height, width, channels))
            Y_test = labels[sample]
            for j,i in enumerate(sample):
                #image = cv2.imread(path + "%s.png" % str(i))
                image = Image.open(path + "%s.png" % str(i))
                #image = resize(image, (height, width, channels), mode='reflect',  preserve_range=True)
                image = np.array(image)
                X_test[j] = image

            steps = index_train.shape[0]/self.batch_size
            model.fit_generator(mu.generator(index_train, self.batch_size, steps, dims=self.dims),
                                steps_per_epoch=steps, epochs=self.epochs, verbose=1, max_queue_size=4)

            # Predict test set and obtain probabilities
            probs = model.predict(X_test, batch_size=self.batch_size, verbose=1)
            preds = np.zeros((32,))
            trues = np.zeros((32,))
            probas = np.zeros((32,))
            for j,i in enumerate(range(0, 512, 16)):
                preds[j] = 1 if np.mean(probs[i:i+16]) >= 0.5 else 0
                trues[j]  = Y_test[i]
                probas[j] = np.mean(probs[i:i+16])

            accuracy = (preds == trues).sum()/trues.shape[0]

            with open(r'predictions_bbbc014_%s' % self.cnn_model, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["Testset accuracy = %s" %accuracy])
                writer.writerow(["Prediciton (first row) vs True (second row):"])
                writer.writerow(probas)
                writer.writerow(trues)

            filename = "saved_bbbc014_{}.h5".format(self.cnn_model)
            model.save(filename)

        return

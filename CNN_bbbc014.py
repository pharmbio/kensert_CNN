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
from json import dumps
from os.path import join

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
                 images_path = "",
                 sample_length = 1024,
                 train_split = 51/64):

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
        self.sample_length = sample_length
        self.train_split = train_split

    def save_model(self, model_path, include_opt=True):
        self.model.save(model_path, include_optimizer=include_opt)

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

        sample_all  = np.array(range(self.sample_length))

        # define range for ~80% of 512 samples
        train_range = int(self.train_split*self.sample_length/2)

        # split samples into two chunks
        sample_MCF7 = np.array(range(0, train_range))
        sample_A549 = np.array(range(int(self.sample_length/2), int(self.sample_length/2)+train_range))

        # index the remaining ~20% of images used for validation
        validation_set = np.setdiff1d(np.setdiff1d(sample_all, sample_A549), sample_MCF7)
        print("Training on {} images, MCF7 indexed {} to {} ({}) and A549 indexed {} to {} ({})".format(
            sample_all.shape[0], sample_MCF7[0], sample_MCF7[-1], len(sample_MCF7), sample_A549[0], sample_A549[-1], len(sample_A549)
        ))

        # Two training sessions, one with MCF7 cell line, and one with A549 cell line.
        for sample in (sample_A549, "A549_{}".format(self.cnn_model)), (sample_MCF7, "MCF7_{}".format(self.cnn_model)):
            self.extend_model()
            index_train = np.array([x for x in sample_all if not (x in sample[0] or x in validation_set)])
            # Compile model, with SGD optimizer
            sgd = SGD(lr=self.lr, momentum=self.momentum)
            self.model.compile(optimizer=sgd, loss="binary_crossentropy", metrics = ["accuracy"])

            steps = index_train.shape[0]/self.batch_size
            self.model.fit_generator(mu.generator(index_train, self.batch_size, steps, dims=self.dims),
                                steps_per_epoch=steps, epochs=self.epochs, verbose=1, max_queue_size=4)

            self.save_model("{}_saved_model.h5".format(sample[1]), include_opt=False)


    def eval_model(self, result_files="./"):
        # load labels; 512 labels, first 256 instances are MCF7 cell lines, second 256 instances are A549 cell lines.
        labels = np.load("bbbc014_labels.npy")

        # define range for ~80% of 512 samples
        train_range = int(self.train_split*self.sample_length/2)

        # split evaluation set into two chunks
        eval_MCF7 = np.array(range(train_range, int(self.sample_length/2)))
        eval_A549 = np.array(range(int(self.sample_length/2)+train_range, self.sample_length))

        height, width, channels = self.dims

        # Two training sessions, one with MCF7 cell line, and one with A549 cell line.
        for evaluation_set in (eval_A549, "A549_{}".format(self.cnn_model)), (eval_MCF7, "MCF7_{}".format(self.cnn_model)):
            self.model = load_model("{}_saved_model.h5".format(evaluation_set[1]), compile=False)
            # Create test set; ith position in labels correspond to ith image.
            X_test = np.zeros((evaluation_set[0].shape[0], height, width, channels))
            Y_test = labels[evaluation_set[0]]

            for j,i in enumerate(evaluation_set[0]):
                image = Image.open(self.path + "%s.png" % str(i))
                image = np.array(image)
                X_test[j] = image

            # Predict test set and obtain probabilities
            probs = self.model.predict(X_test, batch_size=self.batch_size, verbose=1)
            preds = np.zeros((32,))
            trues = np.zeros((32,))
            probas = np.zeros((32,))
            for j,i in enumerate(range(0, evaluation_set[0].shape[0], 16)):
                preds[j] = 1 if np.mean(probs[i:i+16]) >= 0.5 else 0
                trues[j]  = Y_test[i]
                probas[j] = np.mean(probs[i:i+16])


            conf_matrix = confusion_matrix(trues, preds)
            class_report = classification_report(trues, preds, output_dict=False)
            class_report_dict = classification_report(trues, preds, output_dict=True)

            print('Model {}:\nConfusion matrix & classification report for test data'.format(evaluation_set[1]))
            print(conf_matrix)
            print(class_report)
            with open(join(result_files, "{}_confusion_matrix.txt".format(evaluation_set[1])), "w") as cm_file, \
                 open(join(result_files, "{}_classification_report.json".format(evaluation_set[1])), "w") as cr_file:
                cm_file.writelines(str(conf_matrix))
                cr_file.write(dumps(class_report_dict))

        return

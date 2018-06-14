"""
Deep convolutional neural networks: resnet50, inceptionv3, inception_resnetv2.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils import multi_gpu_model
import Keras_ResNet50
import Keras_Inception_v3
import Keras_Inception_Resnet_v2
import model_utils
import numpy as np
import pandas as pd
import csv
import math

from keras.optimizers import SGD
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras import layers
from keras import regularizers
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.callbacks import LearningRateScheduler
from keras.callbacks import CSVLogger

# learning rate schedule
class CNN_Model(object):

    def __init__(self,
                 cnn_model = "ResNet50",
                 dims  = (224,224,3),
                 classes = 12,
                 regularization=0.0,
                 epochs = 5,
                 batch_size = 32,
                 initial_lr=0.0005,
                 drop_lr=0.5,
                 epochs_drop_lr=1.0,
                 momentum = 0.9,
                 weights = "imagenet",
                 compound = None,
                 gpus=0):

        self.cnn_model = cnn_model
        self.dims  = dims
        self.classes = classes
        self.regularization=regularization
        self.epochs =  epochs
        self.batch_size =batch_size
        self.initial_lr = initial_lr
        self.drop_lr   = drop_lr
        self.epochs_drop_lr = epochs_drop_lr
        self.momentum = momentum
        self.weights = weights
        self.compound = compound
        self.gpus = gpus

    # Define model
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
        x = Dense(self.classes, kernel_regularizer=regularizers.l2(self.regularization), activation='softmax', name='predictions')(x)

        extended_model = Model(inputs=model.input, outputs=x)

        return extended_model


    def compile_model(self):
        # Create model
        model = self.extend_model()
        # Define SGD optimizer; learning rate and decay equals zero because learning rate scheduler is used.
        sgd    = SGD(lr=0, decay=0, momentum=self.momentum)
        # Compile model, with SGD optimizer
        if self.gpus > 0:
            model = multi_gpu_model(model, gpus=self.gpus)

        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ["accuracy"])

        return model

    def fit_model(self, model):

        # Read Y labels, including moa, compound, concentration, well, plate, replicate.
        labels  = pd.read_csv('bbbc021_labels.csv', sep=";")

        with open(r'predictions_%s' % self.cnn_model, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([self.compound, labels['moa'][labels['compound'] == self.compound].iloc[0]])

        # learning rate schedule
        def step_decay(epoch):
            """
            Learning rate schedule.

            """
            initial_lrate = self.initial_lr
            drop = self.drop_lr
            epochs_drop = self.epochs_drop_lr
            lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))

            return lrate

        csv_logger     = CSVLogger(self.cnn_model+'_training.log', append=True, separator=';')
        lrate          = LearningRateScheduler(step_decay, verbose=1)

        callbacks_list = [lrate, csv_logger]
        # Index will point at both the Y_labels as well as the images to be imported in model_utils.generator
        index      = np.where(labels['compound'] != self.compound)[0]
        # Calculate Class_weights
        class_weight = model_utils.class_weights(self.compound)
        # Step size (usually input rows divided by batch size)
        steps      = index.shape[0]/self.batch_size
        # Fit model
        model.fit_generator(model_utils.generator(index, self.classes, self.batch_size, dims=self.dims),
                            steps_per_epoch=steps,
                            class_weight=class_weight,
                            epochs=self.epochs,
                            verbose=1,
                            max_queue_size=4,
                            callbacks=callbacks_list)
        return model

    def predict_model(self, model):

        # Prepare test set
        X_test, Y_test = model_utils.load_test_set(self.compound, dims=self.dims)
        # Predict test set and obtain probabilities
        softmaxProbs = model.predict(X_test, batch_size=self.batch_size)

        # Input probabilities and calculate prediction;
        # first: element wise median of the different fields of view,
        # second: element wise median of the replicates.
        # third: element with highest value is the hardPreds
        hardPreds, softPreds, uniqueConc = model_utils.treatment_prediction(self.compound, softmaxProbs)

        #print("Accuracy = " + str(preds))
        with open(r'predictions_%s' % self.cnn_model, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(uniqueConc)
            #writer.writerow((hardPreds == np.argmax(Y_test[0])))
            for i in range(len(uniqueConc)):
                writer.writerow(softPreds[i])

        return

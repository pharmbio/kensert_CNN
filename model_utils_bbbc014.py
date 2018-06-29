import numpy as np
import pandas as pd
import cv2
from PIL import Image

def generator(index, batch_size, steps, dims):
    height, width, channels = dims
    path = "images_bbbc014/bbbc014_"
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, height, width, channels), dtype="uint8")
    batch_labels = np.zeros((batch_size,), dtype="uint8")
    y_labels = np.load("bbbc014_labels.npy")
    np.random.shuffle(index)

    i = 0
    while True:
        for j in range(batch_size):
            if i == index.shape[0]:
                i = 0
                np.random.shuffle(index)
            # Read and pre process image
            image = Image.open(path + "%s.png" % index[i])

            image = np.array(image)
            batch_features[j] = image
            # Assign label i corresponding to i:th feature, then convert to one-hot.
            batch_labels[j] = y_labels[index[i]]
            i += 1

        yield batch_features, batch_labels

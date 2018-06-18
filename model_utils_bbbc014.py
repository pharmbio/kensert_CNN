import numpy as np
import pandas as pd
import cv2
from PIL import Image

def noisy(image):
    row,col,ch= image.shape
    image = image.astype(np.float32)

    for i in range(ch-1):
        jitter_percent = np.random.choice([0., 0., .25, -.25])
        image[:,:,i] *= (1 - jitter_percent)
    if np.random.choice([0,1]) == 1:
        filter_size = 3
        image = cv2.blur(image,(filter_size,filter_size))
    if np.random.choice([0,1]) == 1:
        var   = 50
        sigma = var**0.5
        mean  = 0
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        image += gauss

    image[image < 0] = 0
    image[image > 255] = 255

    return image

def randomize_func(*args):
    return np.random.choice(args)

def generator(index, batch_size, steps, dims):
    height, width, channels = dims

    nor_0   = lambda x: x
    nor_90  = lambda x: np.rot90(x, k=1)
    nor_180 = lambda x: np.rot90(x, k=2)
    nor_270 = lambda x: np.rot90(x, k=3)

    mir_0   = lambda x: np.fliplr(x)
    mir_90  = lambda x: np.rot90(np.fliplr(x), k=1)
    mir_180 = lambda x: np.rot90(np.fliplr(x), k=2)
    mir_270 = lambda x: np.rot90(np.fliplr(x), k=3)

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
            #image = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            func = randomize_func(nor_0, nor_90, nor_180, nor_270, mir_0, mir_90, mir_180, mir_270)
            image = func(image)
            image = noisy(image)
            batch_features[j] = image
            # Assign label i corresponding to i:th feature, then convert to one-hot.
            batch_labels[j] = y_labels[index[i]]
            i += 1

        yield batch_features, batch_labels

import numpy as np
import pandas as pd
import cv2
from PIL import Image


# Define y-labels and dimensions of the images.
moa_dict = {
            'Actin disruptors': 0,          'Aurora kinase inhibitors': 1, 'Cholesterol-lowering': 2,
            'Eg5 inhibitors': 3,            'Epithelial': 4,               'Kinase inhibitors': 5,
            'Microtubule destabilizers': 6, 'Microtubule stabilizers': 7,  'Protein degradation': 8,
            'Protein synthesis': 9,         'DNA replication': 10,         'DNA damage': 11
            }

labels         = pd.read_csv('bbbc021_labels.csv', sep=";")
path           = 'images_transformed_cropped/bbbc021_'
moa            = np.array(labels['moa'])
compounds      = np.array(labels['compound'])
concentrations = np.array(labels['concentration'])
replicates     = np.array(labels['replicate'])

def class_weights(compound):
    y_train = moa[np.where(compounds != compound)[0]]
    y_train = np.asarray([moa_dict[item] for item in y_train])
    y_train = convert_to_one_hot(y_train, 12)
    weights = np.array(list((y_train.shape[0]/(y_train.shape[1]*np.array([y_train[:,i].sum() for i in range(y_train.shape[1])])))))
    class_weights_dict = {}
    for i in range(len(weights)):
        class_weights_dict[i] = weights[i]
    return class_weights_dict

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y]
    return Y

def smooth_labels(labels, eps):
    n_classes = labels.shape[0]
    return labels * (1 - eps) + (1 - labels) * eps / (n_classes - 1.0)

def load_test_set(compound, dims):
    height, width, channels = dims
    # Prepare test set
    test_index  = np.where(compounds == compound)[0]

    Y_test = moa[test_index]
    Y_test = np.asarray([moa_dict[item] for item in Y_test])
    Y_test = convert_to_one_hot(Y_test, 12)

    X_test = np.zeros((len(test_index), height, width, channels))
    for j,i in enumerate(test_index):
        # reading images with cv2 will make "RGB" into "BGR"
        image_test = cv2.imread(path + "%s.png" % str(i))
        resized_image_test = cv2.resize(image_test, (height, width), interpolation=cv2.INTER_AREA)
        #resized_image_test = resized_image_test.astype(np.float32)
        #resized_image_test /= 127.5
        #resized_image_test -= 1.
        X_test[j] = resized_image_test#cv2.imread(path + "%s.png" % str(i))

    return X_test, Y_test

def noisy(image):
    row,col,ch= image.shape
    image = image.astype(np.float32)

    for i in range(ch):
        shade_percent = np.random.choice([0., 0., .25, -.25])
        image[:,:,i] *= (1 - shade_percent)
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

def img_preprocess(i, index, dims):

    nor_0   = lambda x: x
    nor_90  = lambda x: np.rot90(x, k=1)
    nor_180 = lambda x: np.rot90(x, k=2)
    nor_270 = lambda x: np.rot90(x, k=3)

    mir_0   = lambda x: np.fliplr(x)
    mir_90  = lambda x: np.rot90(np.fliplr(x), k=1)
    mir_180 = lambda x: np.rot90(np.fliplr(x), k=2)
    mir_270 = lambda x: np.rot90(np.fliplr(x), k=3)

    height, width, _ = dims

    image = cv2.imread(path + "%s.png" % index[i])
    resized_image = cv2.resize(image, (height, width), interpolation=cv2.INTER_AREA)
    func = randomize_func(nor_0, nor_90, nor_180, nor_270, mir_0, mir_90, mir_180, mir_270)
    transformed_image = func(resized_image)
    transformed_image = noisy(transformed_image)
    #transformed_image = transformed_image.astype(np.float32)
    #transformed_image /= 127.5
    #transformed_image -= 1.
    return transformed_image

def generator(index, classes, batch_size, dims):
    height, width, channels = dims
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, height, width, channels))
    batch_labels = np.zeros((batch_size, classes))
    np.random.shuffle(index)
    i = 0
    while True:
        for j in range(batch_size):
            if i == index.shape[0]:
                i = 0
                np.random.shuffle(index)
            # Read and pre process image
            transformed_image = img_preprocess(i, index, dims)
            batch_features[j] = transformed_image
            # Assign label i corresponding to i:th index, then convert to one-hot.
            label = moa_dict[labels.iloc[index[i], 3]]
            label = convert_to_one_hot(label, classes)
            #label = smooth_labels(label, 0.1)
            batch_labels[j] = label
            #smooth_labels(label, eps)
            i += 1

        yield batch_features, batch_labels


def treatment_prediction(compound, probs):
    test_index  = np.where(compounds == compound)[0]
    conc_unique = list(set(concentrations[test_index]))
    repl_unique = list(set(replicates[test_index]))

    n_conc = len(conc_unique)
    predictions   = np.zeros((n_conc,), dtype="uint8")
    probabilities = np.zeros((n_conc, 12))

    for i, conc in enumerate(conc_unique):
        n_repl = len(repl_unique)
        sub_preds_median = np.zeros((n_repl, 12))

        for j, repl in enumerate(repl_unique):
            test_repl = np.where((replicates[test_index] == repl) & (concentrations[test_index] == conc))[0]

            if test_repl.shape[0] > 0:
                sub_pred = np.array(probs)[test_repl]
                sub_preds_median[j,:] = np.median(sub_pred, axis=0)

        predictions[i] = np.argmax(np.median(np.array(sub_preds_median), axis=0), axis=0)
        probabilities[i] = np.median(np.array(sub_preds_median), axis=0)

    return predictions, probabilities, conc_unique

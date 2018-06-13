"""
A deep Inception network.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras import layers
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras import regularizers
from keras import backend as K
from keras.utils.data_utils import get_file
import numpy as np

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None,
              regularizer=0.0):

    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        kernel_regularizer=regularizers.l2(regularizer),
        use_bias=False,
        name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = Activation('relu', name=name)(x)
    return x


def InceptionV3(input_shape=(299,299,3),regularizer=0.0, weights=None):
    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 32, 3, 3, padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 64, 3, 3, regularizer=regularizer)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 192, 3, 3, padding='valid', regularizer=regularizer)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0, 1, 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)

    branch5x5 = conv2d_bn(x, 48, 1, 1, regularizer=regularizer)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, regularizer=regularizer)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, regularizer=regularizer)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    # mixed 1: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)

    branch5x5 = conv2d_bn(x, 48, 1, 1, regularizer=regularizer)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, regularizer=regularizer)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, regularizer=regularizer)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    # mixed 2: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)

    branch5x5 = conv2d_bn(x, 48, 1, 1, regularizer=regularizer)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5, regularizer=regularizer)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, regularizer=regularizer)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid', regularizer=regularizer)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, regularizer=regularizer)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid', regularizer=regularizer)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)

    branch7x7 = conv2d_bn(x, 128, 1, 1, regularizer=regularizer)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7, regularizer=regularizer)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, regularizer=regularizer)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, regularizer=regularizer)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, regularizer=regularizer)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)

        branch7x7 = conv2d_bn(x, 160, 1, 1, regularizer=regularizer)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7, regularizer=regularizer)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, regularizer=regularizer)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, regularizer=regularizer)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, regularizer=regularizer)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7, regularizer=regularizer)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1, regularizer=regularizer)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, regularizer=regularizer)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, regularizer=regularizer)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)

    branch7x7 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7, regularizer=regularizer)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1, regularizer=regularizer)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1, regularizer=regularizer)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7, regularizer=regularizer)

    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, regularizer=regularizer)
    x = layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid', regularizer=regularizer)

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, regularizer=regularizer)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7, regularizer=regularizer)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1, regularizer=regularizer)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid', regularizer=regularizer)

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, regularizer=regularizer)

        branch3x3 = conv2d_bn(x, 384, 1, 1, regularizer=regularizer)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3, regularizer=regularizer)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1, regularizer=regularizer)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, regularizer=regularizer)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3, regularizer=regularizer)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3, regularizer=regularizer)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1, regularizer=regularizer)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, regularizer=regularizer)
        x = layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))

    model = Model(img_input, x, name='inception_v3')
    if weights == "imagenet":
        weights_path = get_file(
                'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='bcbd6486424b2319ff4ef7d526e38f63')
        model.load_weights(weights_path)
    else: pass

    return model

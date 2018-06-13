"""
A deep residual network with 50 layers (ResNet50).
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from keras.layers import Input
from keras import layers
from keras import regularizers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils.data_utils import get_file


def identity_block(input_tensor, kernel_size, filters, stage, block, regularizer):

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, kernel_regularizer=regularizers.l2(regularizer), padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, regularizer, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=regularizers.l2(regularizer), name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
def ResNet50(input_shape=(224,224,3), regularizer=0.0001, weights=None):

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    img_input = Input(shape=input_shape)

    x = ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), kernel_regularizer=regularizers.l2(regularizer), padding='valid', name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x,     3, [64, 64, 256], stage=2, block='a', regularizer=regularizer, strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', regularizer=regularizer)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', regularizer=regularizer)

    x = conv_block(x,     3, [128, 128, 512], stage=3, block='a', regularizer=regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', regularizer=regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', regularizer=regularizer)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', regularizer=regularizer)

    x = conv_block(x,     3, [256, 256, 1024], stage=4, block='a', regularizer=regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', regularizer=regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', regularizer=regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', regularizer=regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', regularizer=regularizer)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', regularizer=regularizer)

    ###
    x = conv_block(x,     3, [512, 512, 2048], stage=5, block='a', regularizer=regularizer)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', regularizer=regularizer)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', regularizer=regularizer)

    model = Model(img_input, x, name='keras_resnet50')
    if weights == "imagenet":
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 WEIGHTS_PATH_NO_TOP,
                                 cache_subdir='models',
                                 md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
    else: pass

    return model

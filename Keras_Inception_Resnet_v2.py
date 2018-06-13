"""
A deep Inception-Residual network.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras import regularizers
from keras.utils.data_utils import get_file
from keras import backend as K

def conv2d_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              regularizer=0.0,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               kernel_regularizer=regularizers.l2(regularizer),
               name=name)(x)
    if not use_bias:
        bn_axis = 1 if K.image_data_format() == 'channels_first' else 3
        bn_name = None if name is None else name + '_bn'
        x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', regularizer=0.0):

    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(x, 32, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(branch_1, 32, 3, regularizer=regularizer)
        branch_2 = conv2d_bn(x, 32, 1, regularizer=regularizer)
        branch_2 = conv2d_bn(branch_2, 48, 3, regularizer=regularizer)
        branch_2 = conv2d_bn(branch_2, 64, 3, regularizer=regularizer)
        branches = [branch_0, branch_1, branch_2]
    elif block_type == 'block17':
        branch_0 = conv2d_bn(x, 192, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(x, 128, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(branch_1, 160, [1, 7], regularizer=regularizer)
        branch_1 = conv2d_bn(branch_1, 192, [7, 1], regularizer=regularizer)
        branches = [branch_0, branch_1]
    elif block_type == 'block8':
        branch_0 = conv2d_bn(x, 192, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(x, 192, 1, regularizer=regularizer)
        branch_1 = conv2d_bn(branch_1, 224, [1, 3], regularizer=regularizer)
        branch_1 = conv2d_bn(branch_1, 256, [3, 1], regularizer=regularizer)
        branches = [branch_0, branch_1]
    else:
        raise ValueError('Unknown Inception-ResNet block type. '
                         'Expects "block35", "block17" or "block8", '
                         'but got: ' + str(block_type))

    block_name = block_type + '_' + str(block_idx)
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
    up = conv2d_bn(mixed,
                   K.int_shape(x)[channel_axis],
                   1,
                   activation=None,
                   use_bias=True,
                   regularizer=regularizer,
                   name=block_name + '_conv')

    x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
               output_shape=K.int_shape(x)[1:],
               arguments={'scale': scale},
               name=block_name)([x, up])
    if activation is not None:
        x = Activation(activation, name=block_name + '_ac')(x)
    return x

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'
def InceptionResNetV2(input_shape=(299,299,3), regularizer=0.0, weights=None):

    img_input = Input(shape=input_shape)

    # Stem block: 35 x 35 x 192
    x = conv2d_bn(img_input, 32, 3, strides=2, padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 32, 3, padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 64, 3, regularizer=regularizer)
    x = MaxPooling2D(3, strides=2)(x)
    x = conv2d_bn(x, 80, 1, padding='valid', regularizer=regularizer)
    x = conv2d_bn(x, 192, 3, padding='valid', regularizer=regularizer)
    x = MaxPooling2D(3, strides=2)(x)

    # Mixed 5b (Inception-A block): 35 x 35 x 320
    branch_0 = conv2d_bn(x, 96, 1, regularizer=regularizer)
    branch_1 = conv2d_bn(x, 48, 1, regularizer=regularizer)
    branch_1 = conv2d_bn(branch_1, 64, 5, regularizer=regularizer)
    branch_2 = conv2d_bn(x, 64, 1, regularizer=regularizer)
    branch_2 = conv2d_bn(branch_2, 96, 3, regularizer=regularizer)
    branch_2 = conv2d_bn(branch_2, 96, 3, regularizer=regularizer)
    branch_pool = AveragePooling2D(3, strides=1, padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, regularizer=regularizer)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    channel_axis = 1 if K.image_data_format() == 'channels_first' else 3
    x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

    # 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
    for block_idx in range(1, 11):
        x = inception_resnet_block(x,
                                   scale=0.17,
                                   block_type='block35',
                                   block_idx=block_idx,
                                   regularizer=regularizer)

    # Mixed 6a (Reduction-A block): 17 x 17 x 1088
    branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid', regularizer=regularizer)
    branch_1 = conv2d_bn(x, 256, 1, regularizer=regularizer)
    branch_1 = conv2d_bn(branch_1, 256, 3, regularizer=regularizer)
    branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid', regularizer=regularizer)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

     # 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
    for block_idx in range(1, 21):
        x = inception_resnet_block(x,
                                   scale=0.1,
                                   block_type='block17',
                                   block_idx=block_idx,
                                   regularizer=regularizer)

    # Mixed 7a (Reduction-B block): 8 x 8 x 2080
    branch_0 = conv2d_bn(x, 256, 1, regularizer=regularizer)
    branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid', regularizer=regularizer)
    branch_1 = conv2d_bn(x, 256, 1, regularizer=regularizer)
    branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid', regularizer=regularizer)
    branch_2 = conv2d_bn(x, 256, 1, regularizer=regularizer)
    branch_2 = conv2d_bn(branch_2, 288, 3, regularizer=regularizer)
    branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid', regularizer=regularizer)
    branch_pool = MaxPooling2D(3, strides=2, padding='valid')(x)
    branches = [branch_0, branch_1, branch_2, branch_pool]
    x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

    # 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
    for block_idx in range(1, 10):
        x = inception_resnet_block(x,
                                   scale=0.2,
                                   block_type='block8',
                                   block_idx=block_idx,
                                   regularizer=regularizer)
    x = inception_resnet_block(x,
                               scale=1.,
                               activation=None,
                               block_type='block8',
                               block_idx=10,
                               regularizer=regularizer)

    # Final convolution block: 8 x 8 x 1536
    x = conv2d_bn(x, 1536, 1, name='conv_7b', regularizer=regularizer)


    # Create model
    model = Model(img_input, x, name='inception_resnet_v2')

    if weights == "imagenet":
        fname = 'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file(fname,
                                BASE_WEIGHT_URL + fname,
                                cache_subdir='models',
                                file_hash='d19885ff4a710c122648d3b5c3b684e4')

        model.load_weights(weights_path)
    else: pass

    return model

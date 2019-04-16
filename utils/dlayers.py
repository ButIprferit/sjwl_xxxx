from keras.layers import *
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, add, Permute, Conv2D
from keras import backend as K


'''
This script have defin some funcation of layers
'''

from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np
from keras.regularizers import l2
class ScaledLayer(Layer):
    # a scaled layer
    # after this layer can used softmax layer equal to tempature softmax
    def __init__(self, **kwargs):
        super(ScaledLayer
    , self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True,regularizer=l2(0.001))
        super(ScaledLayer
    , self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        return tf.multiply(x, self.W)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)


def BatchNormal(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def squeeze_excite_block(input, ratio=16):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x


def squeeze_excite_block(input, ratio=16,pre_name=None):
    ''' Create a channel-wise squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    '''
    init = input
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(name=pre_name+'globalpooling')(init)
    se = Reshape(se_shape,name=pre_name+'reshape')(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False,name=pre_name+'dense1')(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False,name=pre_name+'dense2')(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x



def channel_spatial_squeeze_excite(input, ratio=16):
    ''' Create a spatial squeeze-excite block
    Args:
        input: input tensor
        filters: number of output filters
    Returns: a keras tensor
    References
    -   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
    -   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
    '''

    cse = squeeze_excite_block(input, ratio)
    sse = spatial_squeeze_excite_block(input)

    x = add([cse, sse])
    return x

class MYLAYER(Layer):
    def __init__(self,**kwargs):
        super(MYLAYER, self).__init__(**kwargs)
    def build(self, input_shape):
        self.w=self.add_weight(name='tempature',
                                       shape=[1],
                                       initializer='ones',
                                       trainable=True)
        self.b=self.add_weight(name='tempature',
                                       shape=[1],
                                       initializer='zeros',
                                       trainable=True)
        super(MYLAYER,self).build(input_shape)
    def call(self, inputs, **kwargs):
        return tf.add(tf.multiply(inputs, self.w),self.b)
    def compute_output_shape(self, input_shape):
        return input_shape
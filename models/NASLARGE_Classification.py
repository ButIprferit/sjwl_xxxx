from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings
from keras.backend import tensorflow_backend as backend
from keras_applications import correct_pad
from keras_applications import get_submodules_from_kwargs
from keras_applications import imagenet_utils
from keras import layers

from keras.layers import Dense,Dropout,LeakyReLU,Activation
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.applications import NASNetLarge,InceptionResNetV2,DenseNet201,NASNetMobile,Xception
from keras import Input,Model
from keras.layers import Dense,Dropout,BatchNormalization,Activation
from keras.regularizers import l2
import keras.utils
from utils.dlayers import *
from spp.SpatialPyramidPooling import SpatialPyramidPooling
from keras_applications.nasnet import _reduction_a_cell,_normal_a_cell
BASE_WEIGHTS_PATH = ('https://github.com/titu1994/Keras-NASNet/'
                     'releases/download/v1.2/')

NASNET_MOBILE_WEIGHT_PATH = BASE_WEIGHTS_PATH + 'NASNet-mobile.h5'
NASNET_MOBILE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-mobile-no-top.h5'
NASNET_LARGE_WEIGHT_PATH = BASE_WEIGHTS_PATH + 'NASNet-large.h5'
NASNET_LARGE_WEIGHT_PATH_NO_TOP = BASE_WEIGHTS_PATH + 'NASNet-large-no-top.h5'

# def NASLARGEClass(input_shape=(762//2,1024//2,3),classes=21,droprate=0.4,kernel_regu_rate=0.00001):
#     naslarge_model = NASNetLarge(include_top=False,
#                               weights='imagenet',
#                               input_tensor=None,
#                               input_shape=input_shape,
#                               pooling='max',
#                               classes=1000)
#     naslarge_model.trainable = False
#     x=naslarge_model.outputs
#     x=Dropout(droprate)(x)
#     x=Dense(512, name='dense2048-512')(x)
#     x = BatchNormal(x)
#     x=Dropout(droprate)(x)
#     x=Dense(128, name='dense512-128')(x)
#     x=BatchNormal(x)
#     x=Dense(classes, activation='softmax', name='out')(x)
#     model=Model(naslarge_model.inputs,x)
#     return model
import keras.utils as keras_utils
backend = keras.backend
layers =keras.layers
models =keras.models



def NASNetClass(input_shape=(331,331,3),
           penultimate_filters=4032,
           num_blocks=6,
           stem_block_filters=96,
           skip_reduction=True,
           filter_multiplier=2,
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           classes=1000,
           default_size=None,
           kernel_regu=0.000001,
           droprate=0.4,
           **kwargs):
    '''Instantiates a NASNet model.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        input_shape: Optional shape tuple, the input shape
            is by default `(331, 331, 3)` for NASNetLarge and
            `(224, 224, 3)` for NASNetMobile.
            It should have exactly 3 input channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        penultimate_filters: Number of filters in the penultimate layer.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        num_blocks: Number of repeated blocks of the NASNet model.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        stem_block_filters: Number of filters in the initial stem block
        skip_reduction: Whether to skip the reduction step at the tail
            end of the network.
        filter_multiplier: Controls the width of the network.
            - If `filter_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `filter_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `filter_multiplier` = 1, default number of filters from the
                 paper are used at each layer.
        include_top: Whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: Optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: Optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: Specifies the default image size of the model

    # Returns
        A Keras model instance.

    # Raises
        ValueError: In case of invalid argument for `weights`,
            invalid input shape or invalid `penultimate_filters` value.
    '''
    global backend,layers,models,keras_utils
    # backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')


    if (isinstance(input_shape, tuple) and
            None in input_shape and
            weights == 'imagenet'):
        raise ValueError('When specifying the input shape of a NASNet'
                         ' and loading `ImageNet` weights, '
                         'the input_shape argument must be static '
                         '(no None entries). Got: `input_shape=' +
                         str(input_shape) + '`.')

    if default_size is None:
        default_size = 331

    # Determine proper input shape and default size.
    # input_shape = _obtain_input_shape(input_shape,
    #                                   default_size=default_size,
    #                                   min_size=32,
    #                                   data_format=backend.image_data_format(),
    #                                   require_flatten=include_top,
    #                                   weights=weights)


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if penultimate_filters % 24 != 0:
        raise ValueError(
            'For NASNet-A models, the value of `penultimate_filters` '
            'needs to be divisible by 24. Current value: %d' %
            penultimate_filters)

    channel_dim =-1
    filters = penultimate_filters // 24

    x = layers.Conv2D(stem_block_filters, (3, 3),
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='stem_conv1',
                      kernel_initializer='he_normal')(img_input)

    x = layers.BatchNormalization(
        axis=channel_dim, momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

    p = None
    x, p = _reduction_a_cell(x, p, filters // (filter_multiplier ** 2),
                             block_id='stem_1')
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier,
                             block_id='stem_2')

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id='%d' % (i))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier,
                              block_id='reduce_%d' % (num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier,
                              block_id='%d' % (num_blocks + i + 1))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier ** 2,
                              block_id='reduce_%d' % (2 * num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier ** 2,
                              block_id='%d' % (2 * num_blocks + i + 1))

    x = layers.Activation('relu')(x)


    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    x=Conv2D(2048,kernel_size=3,strides=2,padding='same',name='stride2-conv',kernel_regularizer=l2(kernel_regu))(x)
    x=squeeze_excite_block(x,16,pre_name='se1')
    x=SpatialPyramidPooling([1,2])(x)
    x=Dense(512,use_bias=True,kernel_regularizer=l2(kernel_regu),name='dense512')(x)
    x=BatchNormalization(name='bn-512')(x)
    x=LeakyReLU(name='leaky-512')(x)
    x=Dropout(droprate,name='drop-512')(x)
    x=Dense(128,kernel_regularizer=l2(kernel_regu),name='dense128')(x)
    x=BatchNormalization(name='batch-128')(x)
    x=LeakyReLU(name='leaky-128')(x)
    x=Dropout(droprate)(x)
    x=Dense(classes,name='out')(x)
    x=MYLAYER(name='scale-layer')(x)
    x=Activation('softmax',name='softmax-classes')(x)

    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = models.Model(inputs, x, name='NASNet')

    # Load weights.
    if weights == 'imagenet':
        weights_path = keras_utils.get_file(
            'nasnet_large_no_top.h5',
            NASNET_LARGE_WEIGHT_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='d81d89dc07e6e56530c4e77faddd61b5')
        model.load_weights(weights_path,by_name=True,skip_mismatch=True)
    else:
        model.load_weights(weights)
    for i in range(1039):
        model.layers[i].trainable = False
    return model



if __name__=='__main__':
    # model=NASNet(input_shape=(762//2,1024//2,3))
    # model.summary()
    pass
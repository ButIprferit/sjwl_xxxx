from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import numpy as np

class ScaledLayer(Layer):
    # a scaled layer
    # after this layer can used softmax layer equal to tempature softmax
    def __init__(self, **kwargs):
        super(ScaledLayer
    , self).__init__(**kwargs)
    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.W = self.add_weight(shape=(1,), # Create a trainable weight variable for this layer.
                                 initializer='one', trainable=True)
        super(ScaledLayer
    , self).build(input_shape)  # Be sure to call this somewhere!
    def call(self, x, mask=None):
        return tf.multiply(x, self.W)
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

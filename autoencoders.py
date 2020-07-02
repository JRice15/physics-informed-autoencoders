import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Lambda, Layer, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Softmax, Subtract, UpSampling2D,
                          ZeroPadding2D, add)
from keras.models import Model
from keras.activations import tanh

from regularizers import *
from common import *

"""
generic autoencoder building blocks to be used between implementations
"""



class FullyConnectedBlock(Layer):
    """
    Dense layer with optional TanH activation and/or Batchnorm
    """

    def __init__(self, name, output_dims, activate=True, batchnorm=False):
        super().__init__()
        self.layer_name = name
        self.layer_flags = (activate, batchnorm)

        self.dense = Dense(output_dims,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            name=name+"-dense"
        )
        self.activation = None
        self.batchnorm = None
        if activate:
            self.activation = Activation(tanh, name=name+"-tanh")
        if batchnorm:
            self.batchnorm = BatchNormalization(name=name+"-batchnorm")
    
    def call(self, inputs):
        x = self.dense(inputs)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x

    def get_config(self):
        return {
            "name": self.layer_name,
            "output_dims": self.dense.units,
            "activate": self.layer_flags[0],
            "batchnorm": self.layer_flags[1],
        }



class AutoencoderBlock(Layer):
    """
    block of 3 fully connected layers, either for encoder or decoder
    """

    def __init__(self, sizes, name, activate_last=False, batchnorm_last=False):
        super().__init__()
        self.sizes = sizes
        self.layer_name = name
        self.activate = activate_last
        self.batchnorm = batchnorm_last

        self.block1 = FullyConnectedBlock(name+"1", sizes[0])
        self.block2 = FullyConnectedBlock(name+"1", sizes[1])
        self.block3 = FullyConnectedBlock(name+"1", sizes[2], activate=activate_last,
            batchnorm=batchnorm_last)

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def get_config(self):
        return {
            "sizes": self.sizes,
            "name": self.layer_name,
            "activate_last": self.activate,
            "batchnorm_last": self.batchnorm,
        }


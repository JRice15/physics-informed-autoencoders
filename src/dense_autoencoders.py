import abc
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, regularizers
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                          Layer, LeakyReLU, MaxPooling2D, PReLU, ReLU, Reshape,
                          Softmax, Subtract, UpSampling2D, ZeroPadding2D, add)
from keras.models import Model

from src.common import *
from src.read_dataset import *


class FullyConnectedBlock(Layer):
    """
    Dense layer with optional TanH activation and/or Batchnorm
    """

    def __init__(self, name, output_dims, weight_decay, activate=True, 
            batchnorm=False, activation_name=None, activation=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dims = output_dims
        self.weight_decay = weight_decay
        self.activation_name = activation

        self.dense = Dense(output_dims,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            kernel_regularizer=regularizers.l2(weight_decay), # weight decay
            name=name+"-dense"
        )
        self.activation = None
        self.batchnorm = None
        if activate:
            self.activation = get_activation(activation, name=name)
        if batchnorm:
            self.batchnorm = BatchNormalization(name=name+"-batchnorm")
    
    def call(self, x):
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dims": self.output_dims,
            "weight_decay": self.weight_decay,
            "activate": self.activation is not None,
            "batchnorm": self.batchnorm is not None,
            "activation_name": self.activation_name,
        })
        return config


class DenseAutoencoderBlock(Layer):
    """
    block of 3 fully connected layers, either for encoder or decoder
    """

    def __init__(self, sizes, weight_decay, name, activate_last=False, 
            activation="tanh", batchnorm_last=False, mask=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.sizes = sizes
        self.weight_decay = weight_decay
        self.activate_last = activate_last
        self.batchnorm_last = batchnorm_last
        self.activation_name = activation
        self.mask = mask

        self.block1 = FullyConnectedBlock(name+"1", sizes[0], weight_decay, activation=activation)
        self.block2 = FullyConnectedBlock(name+"2", sizes[1], weight_decay, activation=activation)
        self.block3 = FullyConnectedBlock(name+"3", sizes[2], weight_decay,
            activate=activate_last, batchnorm=batchnorm_last, activation=activation)
        if self.mask is not None:
            self.mask_layer = LandMask(self.mask)

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if self.mask is not None:
            x = self.mask_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "sizes": self.sizes,
            "weight_decay": self.weight_decay,
            "activate_last": self.activate_last,
            "batchnorm_last": self.batchnorm_last,
            "activation": self.activation_name,
            "mask": self.mask,
        })
        return config


CUSTOM_OBJ_DICT.update({
    "DenseAutoencoderBlock": DenseAutoencoderBlock,
    "FullyConnectedBlock": FullyConnectedBlock
})

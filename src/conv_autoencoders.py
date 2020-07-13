import sys
import abc

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, regularizers
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                          Layer, LeakyReLU, MaxPooling2D, ReLU, Reshape,
                          Softmax, Subtract, UpSampling2D, ZeroPadding2D, add)
from keras.models import Model

from src.common import *
from src.read_dataset import *



class ConvBlock(Layer):
    """
    Conv layer and Pooling, with optional TanH activation and/or Batchnorm
    """

    def __init__(self, name, kernel_size, pool_size, weight_decay, activate=True, 
            batchnorm=False, enc=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.conv = Conv2D(
            filters=1,
            kernel_size=kernel_size,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            kernel_regularizer=regularizers.l2(weight_decay), # weight decay
            name=name+"-conv"
        )
        if enc:
            self.size_change = MaxPooling2D(pool_size=pool_size, name=name+"-maxpool")
        else:
            self.size_change = UpSampling2D(size=pool_size, name=name+"-upsampling")
        self.activation = None
        self.batchnorm = None
        if activate:
            self.activation = Activation(tanh, name=name+"-tanh")
        if batchnorm:
            self.batchnorm = BatchNormalization(name=name+"-batchnorm")
    
    def call(self, x):
        x = self.conv(x)
        x = self.size_change(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "weight_decay": self.weight_decay,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "activate": self.activation is not None,
            "batchnorm": self.batchnorm is not None,
        })
        return config


class ConvAutoencoderBlock(Layer):
    """
    block of 3 conv layers, either for encoder or decoder
    """

    def __init__(self, pool_sizes, kernel_sizes, weight_decay, name, activate_last=False, 
            batchnorm_last=False, encoder=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.pool_sizes = pool_sizes
        self.kernel_sizes = kernel_sizes
        self.weight_decay = weight_decay
        self.activate_last = activate_last
        self.batchnorm_last = batchnorm_last

        self.block1 = ConvBlock(name+"1", kernel_sizes[0], pool_sizes[0], weight_decay, enc=encoder)
        self.block2 = ConvBlock(name+"2", kernel_sizes[1], pool_sizes[1], weight_decay, enc=encoder)
        self.block3 = ConvBlock(name+"3", kernel_sizes[2], pool_sizes[2], weight_decay, enc=encoder,
            activate=activate_last, batchnorm=batchnorm_last)

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_sizes": self.pool_sizes,
            "kernel_sizes": self.kernel_sizes,
            "weight_decay": self.weight_decay,
            "activate_last": self.activate_last,
            "batchnorm_last": self.batchnorm_last,
        })
        return config

CUSTOM_OBJ_DICT.update({
    "ConvAutoencoderBlock": ConvAutoencoderBlock,
    "ConvBlock": ConvBlock
})

import abc
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, regularizers
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Conv2DTranspose, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Lambda,
                          Layer, LeakyReLU, MaxPooling2D, ReLU, Reshape,
                          Softmax, Subtract, UpSampling2D, ZeroPadding2D, add)
from keras.models import Model

from src.common import *
from src.read_dataset import *



class CropToTarget(Layer):

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.do_crop = True
    
    def build(self, input_shape):
        cropx = input_shape[1] - self.target_shape[0]
        cropy = input_shape[2] - self.target_shape[1]
        if cropx == 0 and cropy == 0:
            self.do_crop = False
        crop = [[cropx//2, (cropx+1)//2], [cropy//2, (cropy+1)//2]]
        self.crop =  Cropping2D(crop)

    def call(self, x):
        if self.do_crop:
            return self.crop(x)
        else:
            return x


class ConvBlock(Layer):
    """
    Conv layer and Pooling, with optional TanH activation and/or Batchnorm
    """

    def __init__(self, name, kernel_size, stride, weight_decay, activate=True, 
            batchnorm=False, enc=True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.stride = stride

        conv_layer = Conv2D if enc else Conv2DTranspose
        self.conv = conv_layer(
            filters=1,
            kernel_size=kernel_size,
            strides=stride,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            kernel_regularizer=regularizers.l2(weight_decay), # weight decay
            name=name+"-conv"
        )
        self.activation = None
        self.batchnorm = None
        if activate:
            self.activation = Activation(tanh, name=name+"-tanh")
        if batchnorm:
            self.batchnorm = BatchNormalization(name=name+"-batchnorm")
    
    def call(self, x):
        x = self.conv(x)
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
            "stride": self.stride,
            "activate": self.activation is not None,
            "batchnorm": self.batchnorm is not None,
        })
        return config


class ConvAutoencoderBlock(Layer):
    """
    block of 3 conv layers, either for encoder or decoder
    Args:
        encoder (bool): whether this is an encoder (ie, false for decoder)
        target_shape: only required for decoder, to get proper output shape
    """

    def __init__(self, strides, kernel_sizes, weight_decay, name, activate_last=False, 
            batchnorm_last=False, encoder=True, target_shape=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.is_encoder = encoder
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.weight_decay = weight_decay
        self.activate_last = activate_last
        self.batchnorm_last = batchnorm_last
        self.target_shape = target_shape

        self.block1 = ConvBlock(name+"1", kernel_sizes[0], strides[0], weight_decay, enc=encoder)
        self.block2 = ConvBlock(name+"2", kernel_sizes[1], strides[1], weight_decay, enc=encoder)
        self.block3 = ConvBlock(name+"3", kernel_sizes[2], strides[2], weight_decay, enc=encoder,
            activate=activate_last, batchnorm=batchnorm_last)

    def call(self, x):
        x = AddChannels()(x)
        if not self.is_encoder:
            # add 1 row and column so that we end up with shape larger than or 
            # equal to the target
            x = ZeroPadding2D([[0,1],[0,1]])(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        if not self.is_encoder:
            # get rid of any extra rows/cols in as even a way possible
            x = CropToTarget(self.target_shape)(x)
        x = RemoveChannels()(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "encoder": self.is_encoder,
            "strides": self.strides,
            "target_shape": self.target_shape,
            "kernel_sizes": self.kernel_sizes,
            "weight_decay": self.weight_decay,
            "activate_last": self.activate_last,
            "batchnorm_last": self.batchnorm_last,
        })
        return config

CUSTOM_OBJ_DICT.update({
    "ConvAutoencoderBlock": ConvAutoencoderBlock,
    "ConvBlock": ConvBlock,
    "CropToTarget": CropToTarget,
})

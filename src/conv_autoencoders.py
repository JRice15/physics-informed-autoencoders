import abc
import sys

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input, regularizers
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Conv2DTranspose, Cropping2D, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Lambda, Layer, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Softmax, Subtract, UpSampling2D,
                          ZeroPadding2D, add)
from keras.models import Model

from src.common import *
from src.read_dataset import *


class CropToTarget(Layer):
    """
    Args:
        target_shape: shape, with no batch size
    """

    def __init__(self, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.do_crop = True
    
    def build(self, input_shape):
        cropy = input_shape[1] - self.target_shape[0]
        cropx = input_shape[2] - self.target_shape[1]
        if cropx == 0 and cropy == 0:
            self.do_crop = False
        self.crop =  Cropping2D([[0,cropy],[0,cropx]])

    def call(self, x):
        if self.do_crop:
            return self.crop(x)
        else:
            return x


class ConvDilateLayer(Layer):
    """
    Conv layer and Pooling/Upsampling, with optional TanH activation and/or Batchnorm
    Args:
        dilation (int|float): >1 results in pooling, <1 results in upsampling by 1/dilation
    """

    def __init__(self, name, filters, kernel_size, dilation, weight_decay, activate=True, 
            batchnorm=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.filters = filters

        conv_layer = Conv2D if dilation > 1 else Conv2DTranspose
        self.conv = conv_layer(
            filters=filters,
            kernel_size=kernel_size,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            kernel_regularizer=regularizers.l2(weight_decay), # weight decay
            name=name+"-conv"
        )
        self.activation = Activation(tanh, name=name+"-tanh") if activate else None
        self.batchnorm = BatchNormalization(name=name+"-batchnorm") if batchnorm else None
        if dilation > 1:
            self.dilation_layer = MaxPooling2D(dilation)
        else:
            self.dilation_layer = UpSampling2D(round(1/dilation))
    
    def call(self, x):
        # dilation is split up so as to reduce computation when possible, and skip the 
        # layer when dilation == 1
        if self.dilation < 1:
            x = self.dilation_layer(x)
        x = self.conv(x)
        if self.dilation > 1:
            x = self.dilation_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "weight_decay": self.weight_decay,
            "kernel_size": self.kernel_size,
            "dilation": self.dilation,
            "activate": self.activation is not None,
            "batchnorm": self.batchnorm is not None,
        })
        return config



class ConvAutoencoderBlock(Layer, abc.ABC):
    """
    base class for encoder and decoder conv layers
    """

    def __init__(self, depth, dilations, kernel_sizes, weight_decay, name, activate_last=False, 
            batchnorm_last=False, **kwargs):
        super().__init__(name=name, **kwargs)
        assert len(kernel_sizes) == len(dilations) == depth
        self.depth = depth
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.weight_decay = weight_decay
        self.activate_last = activate_last
        self.batchnorm_last = batchnorm_last

    def make_conv_layers(self):
        # create number of layers equal to depth
        filters = [2**i for i in range(self.depth-1, -1, -1)]
        for i in range(self.depth-1):
            conv = ConvDilateLayer(name=self.name+str(i), filters=filters[i], 
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i], 
                weight_decay=self.weight_decay)
            setattr(self, "block"+str(i), conv)

        i = self.depth - 1
        conv = ConvDilateLayer(name=self.name+str(i), filters=filters[i], kernel_size=self.kernel_sizes[i], 
            dilation=self.dilations[i], weight_decay=self.weight_decay, activate=self.activate_last,
            batchnorm=self.batchnorm_last)
        setattr(self, "block"+str(i), conv)

    def get_config(self):
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "dilations": self.dilations,
            "kernel_sizes": self.kernel_sizes,
            "weight_decay": self.weight_decay,
            "activate_last": self.activate_last,
            "batchnorm_last": self.batchnorm_last,
        })
        return config

    @abc.abstractmethod
    def call(self, x):
        ...


class ConvEncoder(ConvAutoencoderBlock):
    """
    block of 3 conv layers, either for encoder or decoder
    Args:
        encoder (bool): whether this is an encoder (ie, false for decoder)
        target_shape: only required for decoder, to get proper output shape
    """

    def __init__(self, depth, dilations, kernel_sizes, weight_decay, activate_last=False, 
            batchnorm_last=False, **kwargs):
        super().__init__(name="encoder", depth=depth, dilations=dilations, kernel_sizes=kernel_sizes, 
            weight_decay=weight_decay, activate_last=activate_last, batchnorm_last=batchnorm_last, **kwargs)

        self.encoded_shape = None # computed during call
        self.make_conv_layers()

    def call(self, x):
        x = AddChannels()(x)

        # call each dynamically created block
        for i in range(self.depth):
            block = getattr(self, "block"+str(i))
            x = block(x)

        # save encoded shape so decoder knows how to unflatten
        if self.encoded_shape is None:
            self.encoded_shape = x.shape[1:]
        x = Flatten()(x)

        return x


class ConvDecoder(ConvAutoencoderBlock):
    """
    block of 3 conv layers, either for encoder or decoder
    Args:
        encoder (bool): whether this is an encoder (ie, false for decoder)
        target_shape: only required for decoder, to get proper output shape
    """

    def __init__(self, depth, dilations, kernel_sizes, weight_decay, activate_last=False, 
            batchnorm_last=False, target_shape=None, encoded_shape=None, **kwargs):
        super().__init__(name="decoder", depth=depth, dilations=dilations, kernel_sizes=kernel_sizes, 
            weight_decay=weight_decay, activate_last=activate_last, batchnorm_last=batchnorm_last, **kwargs)

        self.target_shape = target_shape
        self.encoded_shape = encoded_shape
        self.make_conv_layers()

    def call(self, x, withshape=False):
        # unflatten (this automatically adds channels)
        x = Reshape(self.encoded_shape)(x)
        # add 1 row and column so that we end up with shape larger than or 
        # equal to the target
        x = ZeroPadding2D([[0,1],[0,1]])(x)

        # call each dynamically created block
        for i in range(self.depth):
            block = getattr(self, "block"+str(i))
            x = block(x)

        # get rid of any extra rows/cols from the zero padding
        x = CropToTarget(self.target_shape)(x)
        x = RemoveChannels()(x)

        return x



CUSTOM_OBJ_DICT.update({
    "ConvAutoencoderBlock": ConvAutoencoderBlock,
    "ConvDecoder": ConvDecoder,
    "ConvEncoder": ConvEncoder,
    "CropToTarget": CropToTarget,
})

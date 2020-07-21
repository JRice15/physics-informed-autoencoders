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
                          ZeroPadding2D, add, SeparableConv2D, AveragePooling2D)
from keras.models import Model

from src.common import *
from src.read_dataset import *


class CropToTarget(Layer):
    """
    Does not affect channels
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
        up (bool): whether is upsampling or downsampling
        dilation (int|float): >1 results in pooling, <1 results in upsampling by 1/dilation
    """

    def __init__(self, up, name, filters, kernel_size, dilation, weight_decay, activate=True, 
            batchnorm=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.up = up
        self.weight_decay = weight_decay
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.filters = filters

        conv_layer = Conv2DTranspose if up else Conv2D
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
        if self.dilation > 1:
            if not up:
                self.dilation_layer = MaxPooling2D(dilation, name=name+"-maxpool")
                # self.dilation_layer = AveragePooling2D(dilation, name=name+"-avgpool")
            else:
                self.dilation_layer = UpSampling2D(dilation, interpolation='bilinear',
                     name=name+"-upsample")
    
    def call(self, x):
        # dilation is split up so as to reduce computation when possible, and skip the 
        # layer when dilation == 1
        x = self.conv(x)
        if not self.up and self.dilation > 1:
            x = self.dilation_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        if self.up and self.dilation > 1:
            x = self.dilation_layer(x)
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
            "up": self.up
        })
        return config

    def vis_repr(self):
        return {
            "filters": self.filters,
            "dilation": self.dilation,
            "kernel_size": self.kernel_size
        }



class ConvAutoencoderBlock(Layer, abc.ABC):
    """
    base class for encoder and decoder conv layers
    """

    def __init__(self, depth, dilations, kernel_sizes, filters, weight_decay, name, conv_dynamics, 
            activate_last=False, batchnorm_last=False, **kwargs):
        super().__init__(name=name, **kwargs)
        assert len(kernel_sizes) == len(dilations) == depth
        self.depth = depth
        self.dilations = dilations
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.weight_decay = weight_decay
        self.activate_last = activate_last
        self.batchnorm_last = batchnorm_last
        self.conv_dynamics = conv_dynamics

    def make_conv_layers(self, up):
        # create first depth-1 layers dynamically
        for i in range(self.depth-1):
            conv = ConvDilateLayer(up=up, name=self.name+str(i), filters=self.filters[i], 
                kernel_size=self.kernel_sizes[i], dilation=self.dilations[i], 
                weight_decay=self.weight_decay)
            setattr(self, "block"+str(i), conv)

        # final layer
        i = self.depth - 1
        conv = ConvDilateLayer(up=up, name=self.name+str(i), filters=1, kernel_size=self.kernel_sizes[i], 
            dilation=self.dilations[i], weight_decay=self.weight_decay, activate=self.activate_last,
            batchnorm=self.batchnorm_last)
        setattr(self, "block"+str(i), conv)

    def get_config(self):
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "dilations": self.dilations,
            "kernel_sizes": self.kernel_sizes,
            "filters": self.filters,
            "weight_decay": self.weight_decay,
            "activate_last": self.activate_last,
            "batchnorm_last": self.batchnorm_last,
            "conv_dynamics": self.conv_dynamics,
        })
        return config

    @abc.abstractmethod
    def call(self, x):
        ...


class ConvEncoder(ConvAutoencoderBlock):
    """
    variable depth convolutional encoder
    """

    def __init__(self, depth, dilations, kernel_sizes, filters, weight_decay, conv_dynamics, activate_last=False, 
            batchnorm_last=False, name="encoder", **kwargs):
        super().__init__(name=name, depth=depth, dilations=dilations, kernel_sizes=kernel_sizes, 
            weight_decay=weight_decay, activate_last=activate_last, batchnorm_last=batchnorm_last, 
            conv_dynamics=conv_dynamics, filters=filters, **kwargs)

        self.encoded_shape = None # computed during call
        self.make_conv_layers(up=False)

        self.add_channels = AddChannels()
        if not self.conv_dynamics:
            self.flattener = Flatten()

    def call(self, x):
        x = self.add_channels(x)

        # call each dynamically created block
        for i in range(self.depth):
            block = getattr(self, "block"+str(i))
            x = block(x)

        # save encoded shape so decoder knows how to unflatten
        if self.encoded_shape is None:
            self.encoded_shape = x.shape[1:]
        if not self.conv_dynamics:
            x = self.flattener(x)

        return x


class ConvDecoder(ConvAutoencoderBlock):
    """
    variable depth convolutional decoder
    """

    def __init__(self, depth, dilations, kernel_sizes, filters, weight_decay, conv_dynamics, activate_last=False, 
            batchnorm_last=False, target_shape=None, encoded_shape=None, name="decoder", **kwargs):
        super().__init__(name=name, depth=depth, dilations=dilations, kernel_sizes=kernel_sizes, 
            weight_decay=weight_decay, activate_last=activate_last, batchnorm_last=batchnorm_last, 
            conv_dynamics=conv_dynamics, filters=filters, **kwargs)

        self.target_shape = target_shape
        self.encoded_shape = encoded_shape
        self.make_conv_layers(up=True)

        if not self.conv_dynamics:
            self.unflatten = Reshape(self.encoded_shape)
        self.pad = ZeroPadding2D([[0,1],[0,1]])
        self.crop = CropToTarget(self.target_shape)
        self.rm_channels = RemoveChannels()
        self.scale = Scale(1.0, name="decoder-scale")

    def call(self, x, withshape=False):
        if not self.conv_dynamics:
            # unflatten (this automatically adds channels)
            x = self.unflatten(x)
        # add 1 row and column so that we end up with shape larger than or 
        # equal to the target
        x = self.pad(x)

        # call each dynamically created block
        for i in range(self.depth):
            block = getattr(self, "block"+str(i))
            x = block(x)

        # get rid of any extra rows/cols from the zero padding
        x = self.crop(x)
        x = self.rm_channels(x)

        # final scaling
        x = self.scale(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "target_shape": self.target_shape,
            "encoded_shape": self.encoded_shape,
        })
        return config



CUSTOM_OBJ_DICT.update({
    "ConvAutoencoderBlock": ConvAutoencoderBlock,
    "ConvDecoder": ConvDecoder,
    "ConvEncoder": ConvEncoder,
    "CropToTarget": CropToTarget,
})

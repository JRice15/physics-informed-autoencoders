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
from src.regularizers import *


"""
generic autoencoder building blocks to be used between implementations
"""



class FullyConnectedBlock(Layer):
    """
    Dense layer with optional TanH activation and/or Batchnorm
    """

    def __init__(self, name, output_dims, weight_decay, activate=True, batchnorm=False):
        super().__init__()

        self.dense = Dense(output_dims,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            kernel_regularizer=regularizers.l2(weight_decay), # weight decay
            name=name+"-dense"
        )
        self.activation = None
        self.batchnorm = None
        if activate:
            self.activation = Activation(tanh, name=name+"-tanh")
        if batchnorm:
            self.batchnorm = BatchNormalization(name=name+"-batchnorm")
    
    def call(self, x):
        x = self.dense(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.batchnorm is not None:
            x = self.batchnorm(x)
        return x


class AutoencoderBlock(Layer):
    """
    block of 3 fully connected layers, either for encoder or decoder
    """

    def __init__(self, sizes, weight_decay, name, activate_last=False, batchnorm_last=False):
        super().__init__()

        self.block1 = FullyConnectedBlock(name+"1", sizes[0], weight_decay)
        self.block2 = FullyConnectedBlock(name+"2", sizes[1], weight_decay)
        self.block3 = FullyConnectedBlock(name+"3", sizes[2], weight_decay, 
            activate=activate_last, batchnorm=batchnorm_last)

    def call(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x



class BaseAE(abc.ABC):
    """
    base container class for autoencoders
    """

    def __init__(self, args):
        self.args = args

    @abc.abstractmethod
    def build_model(self, args):
        ...

    @abc.abstractmethod
    def make_run_name(self):
        ...
    
    @abc.abstractmethod
    def format_data(self, X, Xtest):
        ...

    @abc.abstractmethod
    def train(self, callbacks):
        """
        call model.fit, return History
        """
        ...

    @abc.abstractmethod
    def get_pipeline(self):
        """
        get forward prediction pipeline
        """
        ...

    @abc.abstractmethod
    def save_eigenvals(self):
        ...
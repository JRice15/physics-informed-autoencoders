import math
import sys
import abc

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.models import Model
from keras import Input
from keras.layers import Layer, Dense, Reshape

class AddChannels(Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        # remove batch size, add channels for shape
        self.reshape = Reshape(input_shape[1:] + (1,))

    def call(self, x):
        return self.reshape(x)

class RemoveChannels(Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        # shape with no batch size, and remove channels
        self.reshape = Reshape(input_shape[1:-1])

    def call(self, x):
        return self.reshape(x)

CUSTOM_OBJ_DICT = {
    "AddChannels": AddChannels,
    "RemoveChannels": RemoveChannels
}

class BaseAE(abc.ABC):
    """
    base container class for autoencoders
    """

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def run_name_common_suffix(self):
        args = self.args
        run_name = self.dataset.dataname + "."
        run_name += "{}ep_{}bs_{}lr_{}wd_{}gc.".format(args.epochs, args.batchsize, args.lr, args.wd, args.gradclip)
        if args.convolutional:
            run_name += "k" + "_".join([str(i) for i in args.kernel_sizes])
            run_name += ".s" + "_".join([str(i) for i in args.strides])
        else:
            run_name += "s" + "_".join([str(i) for i in args.sizes])
        run_name += ".{}".format(args.seed)
        return run_name

    @abc.abstractmethod
    def build_enc_dec(self, args, output_dims):
        ...

    @abc.abstractmethod
    def build_model(self, *args):
        ...

    @abc.abstractmethod
    def make_run_name(self):
        ...
    
    @abc.abstractmethod
    def format_data(self, dataset):
        ...

    @abc.abstractmethod
    def compile_model(self, optimizer):
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


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inverse_reg(x, encoder, decoder):
    """
    regularizer to enforce that the decoder is the inverse of the encoder. 
    Equation 9 per Erichson et al.  
    Args:
        encoder, decoder: ComposedLayers
        lambda_: weighting hyperparameter for this loss term
    """
    # q = encoder(x)
    # q_pred = encoder(decoder(q))
    # norm = tf.reduce_mean(tf.square(q - q_pred))
    x_pred = decoder(encoder(x))
    norm = tf.reduce_mean(tf.square(x - x_pred))
    return norm




import abc
import math
import sys
import re

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras import activations
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Conv2DTranspose, Cropping2D, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Lambda, Layer, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Softmax, Subtract, UpSampling2D,
                          ZeroPadding2D, add, PReLU)
from keras.models import Model


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


class Scale(Layer):
    """
    scale output by a single learnable value
    """

    def __init__(self, initial_factor=1.0, use_bias=False, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(initial_factor, (int, float))
        self.initial_factor = float(initial_factor)
        self.use_bias = use_bias

        w_init = lambda shape, dtype=None: self.initial_factor
        self.w = self.add_weight(
            shape=None,
            initializer=w_init,
            trainable=True
        )
        if use_bias:
            b_init = lambda shape, dtype=None: 0.0
            self.b = self.add_weight(
                shape=None,
                initializer=b_init
            )
    
    def call(self, x):
        if self.use_bias:
            return self.w * x + self.b
        return self.w * x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "initial_factor": self.initial_factor
        })
        return config


CUSTOM_OBJ_DICT = {
    "AddChannels": AddChannels,
    "RemoveChannels": RemoveChannels,
    "Scale": Scale,
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
        run_name = ""
        if args.conv_dynamics:
            run_name += "c-dyn."
        run_name += self.dataset.dataname + "."
        run_name += "{}ep_{}bs_{}lr_{}wd_{}gc.".format(args.epochs, args.batchsize, args.lr, args.wd, args.gradclip)
        if args.convolutional:
            run_name += "k" + "".join([str(i) for i in args.kernel_sizes])
            run_name += ".d" + "".join([str(i) for i in args.dilations])
            run_name += ".f" + "".join([str(i) for i in args.filters])
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


def get_activation(act_name, name):
    act_name = act_name.lower().strip()
    act_name = re.sub(r"\s", "", act_name)
    act_name = re.sub(r"[_-]", "", act_name)
    if act_name == "tanh":
        return Activation(activations.tanh, name=name+"-tanh")
    if act_name == "relu":
        return Activation(activations.relu, name=name+"-relu")
    if act_name in ("lrelu", "leakyrelu"):
        return LeakyReLU(0.2, name=name+"-lrelu")
    if act_name in ("prelu", "parametricrelu"):
        return PReLU(name=name+"-prelu")
    if act_name in ("sig", "sigmoid"):
        return Activation(activations.sigmoid, name=name+"-sigmoid")
    if act_name == "softplus":
        return Activation(activations.softplus, name=name+"-softplus")
    
    raise ValueError("Bad activation name '{}'".format(act_name))


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


def vis_model(model: Model):
    print("Model structure:")
    unimportant_names = ["input", "tf_op_layer", "add_loss", "add_metric"]
    for i in model.layers:
        if not any([i.name.startswith(j) for j in unimportant_names]):
            vis_layer(i, 2)


def vis_layer(layer: Layer, indent):
    try:
        rep = layer.vis_repr()
    except AttributeError:
        rep = ""
    print(" "*indent + layer.name + ": " + layer.__class__.__name__ + " " + str(rep))
    for k,v in vars(layer).items():
        if isinstance(v, Layer):
            vis_layer(v, indent+2)


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


class LandMask(Layer):

    def __init__(self, mask, _formatted=False):
        super().__init__()
        if _formatted:
            self.mask = mask
        else:
            mask = tf.convert_to_tensor(mask, K.floatx())
            # flip bits; 0 => 1, 1 => 0
            # now 1's are water, not land
            self.mask = 1 - mask
    
    def call(self, x):
        return x * self.mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask": self.mask,
            "_formatted": True,
        })
        return config



CUSTOM_OBJ_DICT = {
    "AddChannels": AddChannels,
    "RemoveChannels": RemoveChannels,
    "Scale": Scale,
    "LandMask": LandMask,
}

class BaseAE(abc.ABC):
    """
    base container class for autoencoders
    """

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def run_name_common_suffix(self, has_sizes=True):
        args = self.args
        run_name = ""
        if args.conv_dynamics:
            run_name += "c-dyn."
        run_name += self.dataset.dataname + "."
        run_name += get_activation_name(args.activation) + "."
        run_name += "{}ep{}bs{}lr{}wd{}gc".format(args.epochs, args.batchsize, args.lr, args.wd, args.gradclip)
        if args.convolutional:
            run_name += ".k" + "".join([str(i) for i in args.kernel_sizes])
            run_name += ".d" + "".join([str(i) for i in args.dilations])
            run_name += ".f" + "".join([str(i) for i in args.filters])
        elif has_sizes:
            run_name += ".s" + "_".join([str(i) for i in args.sizes])
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
    """
    get activation Layer
    """
    act, _ = _get_activation(act_name, name)
    return act

def get_activation_name(arg_act_name):
    """
    get canonical run_name version of activation name
    """
    _, name = _get_activation(arg_act_name)
    return name

def _get_activation(act_name, name=""):
    """
    helper to get activation and its name for run_names
    Returns:
        Activation, str
    """
    act_name = act_name.lower().strip()
    act_name = re.sub(r"\s", "", act_name)
    act_name = re.sub(r"[_-]", "", act_name)
    if act_name == "tanh":
        return Activation(activations.tanh, name=name+"-tanh"), "tanh"
    if act_name == "relu":
        return Activation(activations.relu, name=name+"-relu"), "relu"
    if act_name in ("lrelu", "leakyrelu"):
        return LeakyReLU(0.2, name=name+"-lrelu"), "lrelu"
    if act_name in ("prelu", "parametricrelu"):
        return PReLU(name=name+"-prelu"), "prelu"
    if act_name in ("sig", "sigmoid"):
        return Activation(activations.sigmoid, name=name+"-sigmoid"), "sig"
    if act_name == "softplus":
        return Activation(activations.softplus, name=name+"-softplus"), "softp"
    
    raise ValueError("Bad activation name '{}'".format(act_name))


def inverse_reg(x, encoder, decoder):
    """
    regularizer to enforce that the decoder is the inverse of the encoder. 
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


def quad_test(x, test, num=4, prnt=True):
    """
    this is a neat little function I made that isn't directly used but is useful
    for visualizing arrays and such it lets you split the array into quadrants 
    (or oct-rants, or anything) and summarizing that quadrant with one value, as
    returned by the callable 'test'. A common test could be: 'lambda x: np.mean(x)'
    """
    if len(x.shape) == 1:
        raise ValueError("x is not 2D")
    rl = len(x) // num
    cl = len(x[0]) // num
    out = []
    for r in range(num):
        row = []
        for c in range(num):
            v = x[ r*rl:(r+1)*rl, c*cl:(c+1)*cl ]
            row.append(test(v))
        out.append(row)
    out = np.array(out)
    if prnt:
        print(out)
    return out

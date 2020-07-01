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

class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.001
    epochs = 6000
    batchsize = 34
    lambda_ = 3 # inverse regularizer weight
    kappa = 3 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes = (40, 25, 15) # largest to smallest

def get_run_name(args):
    run_name = ""
    if args.tag is not None:
        run_name += args.tag + "."
    if args.no_stability:
        run_name += "nostability."
    else:
        run_name += "stability."
    run_name += "{}epochs_{}batchsize_{}lr.".format(args.epochs, args.batchsize, args.lr)
    run_name += "s{}_{}_{}.".format(args.s1, args.s2, args.s3)
    run_name += "l{}_k{}_g{}".format(args.lambd, args.kappa, args.gamma)
    return run_name


class ComposedLayers():
    """
    halfway between a Layer and a Model, because Models don't allow symbolic
    tensors as inputs, and we need to access individual Layer weights for the
    stability regularizer
    """

    def __init__(self, layers):
        self.layers = layers
        self.composed = self._compose_layers(layers)

    def _compose_layers(self, layers):
        """
        turn a list of layers into one callable, where the layers get called in 
        the order of the list
        """
        if len(layers) == 0:
            return lambda x: x
        def f(x):
            return layers[-1]( self._compose_layers(layers[:-1])(x) )
        return f

    def __call__(self, x):
        return self.composed(x)


def make_fc_block(output_dims, name, activate=True, batchnorm=False):
    """
    creates a fully connected layer, with optional tanh activation and batchnorm
    Returns:
        list of Layer
    """
    layers = []
    layers.append(
        Dense(output_dims,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros(),
            name=name+"-dense")
    )
    if activate:
        layers.append( Activation(tanh, name=name+"-tanh") )
    if batchnorm:
        layers.append( BatchNormalization(name=name+"-batchnorm") )
    return layers


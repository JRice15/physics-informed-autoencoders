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

from regularizers import *

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



def shallow_autoencoder(snapshot_shape, output_dims, lambda_, kappa, gamma,
        no_stability=False, sizes=(40,25,15)):
    """
    Create a shallow autoencoder model
    Args:
        snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
        output_dims (int): number of output channels
        lambda_ (float): weighting factor for inverse regularizer
        kappa (float): weighting factor for stability regularizer
        sizes (tuple of int): depth of layers in decreasing order of size. default (40,25,15)
    Returns:
        full: Keras Model for full autoencoder
        encoder: ComposedLayers
        dynamics: Layer
        decoder: ComposedLayers
    """
    # add channels
    inpt = Input(snapshot_shape)
    print("Autoencoder Input shape:", inpt.shape)

    large, medium, small = sizes

    #--------------------------- Encoder --------------------------------------
    encoder_layers = make_fc_block(large, name="enc1")
    encoder_layers += make_fc_block(medium, name="enc2")
    encoder_layers += make_fc_block(small, activate=False, batchnorm=True, name="enc3")

    encoder = ComposedLayers(encoder_layers)
    encoded_output = encoder(inpt)

    #--------------------------- Dynamics -------------------------------------
    dynamics = LyapunovStableDense(kappa=kappa, gamma=gamma, no_stab=no_stability,
        units=small, name="lyapunovstable-dynamics")
    dynamics_output = dynamics(encoded_output)

    #--------------------------- Decoder --------------------------------------
    decoder_layers = make_fc_block(medium, name="dec1")
    decoder_layers += make_fc_block(large, name="dec2")
    decoder_layers += make_fc_block(output_dims, activate=False, name="dec3")

    decoder = ComposedLayers(decoder_layers)
    output = decoder(dynamics_output)

    #--------------------------- Full Model -----------------------------------
    model = Model(inpt, output)

    #--------------------------- Regularization -------------------------------
    # inverse property of encoder-decoder
    inv_loss = lambda_ * inverse_reg(inpt, encoder, decoder)

    model.add_loss(inv_loss)
    model.add_metric(inv_loss, name="inverse", aggregation='mean')

    return model





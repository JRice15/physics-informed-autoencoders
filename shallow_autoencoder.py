import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import Input
from keras.initializers import glorot_normal, zeros
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D, Input,
                          Lambda, Layer, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Softmax, Subtract, UpSampling2D,
                          ZeroPadding2D, add)
from keras.models import Model
from keras.activations import tanh

from regularizers import *


def make_fc_block(output_dims, activate=True, batchnorm=False):
    """
    creates a fully connected layer, with optional tanh activation and batchnorm
    Returns:
        list of Layer
    """
    layers = []
    layers.append(
        Dense(output_dims,
            kernel_initializer=glorot_normal(), # aka Xavier Normal
            bias_initializer=zeros())
    )
    if activate:
        layers.append( Activation(tanh) )
    if batchnorm:
        layers.append( BatchNormalization() )
    return layers

def run_input(layers, x):
    """
    run an input through a list of layers
    """
    for l in layers:
        x = l(x)
    return x


def shallow_autoencoder(input_shape, output_dims, lambda_, sizes=(40,25,15)):
    """
    Create a shallow autoencoder model
    Args:
        input_shape (tuple of int)
        output_dims (int): number of output channels
        lambda_ (float): weighting factor for inverse regularizer
        sizes (tuple of int): depth of layers in decreasing order of size. default (40,25,15)
    Returns:
        4 Models: full, encoder, dynamics, decoder
    """
    inpt = Input(input_shape)
    large, medium, small = sizes

    # Encoder --------------------------------------
    encoder_layers = make_fc_block(large)
    encoder_layers += make_fc_block(medium)
    encoder_layers += make_fc_block(small, activate=False, batchnorm=True)

    encoded_output = run_input(encoder_layers, inpt)
    # extract encoder model
    encoder = Model(inpt, encoded_output)

    # Dynamics -------------------------------------
    dynamics_layers = make_fc_block(small, activate=False)

    dynamics_output = run_input(dynamics_layers, encoded_output)
    # extract dynamics model
    dynamics_input = Input(encoded_output.shape)
    dynamics = Model(dynamics_input, run_input(dynamics_layers, dynamics_input))

    # Decoder --------------------------------------
    decoder_layers = make_fc_block(medium)
    decoder_layers += make_fc_block(large)
    decoder_layers += make_fc_block(output_dims, activate=False)

    output = run_input(decoder_layers, dynamics_output)
    # extract decoder model
    decoder_input = Input(dynamics_output.shape)
    decoder = Model(decoder_input, run_input(decoder_layers, decoder_input))

    # Create full model ----------------------------
    model = Model(inpt, output)

    # regularize inverse property of encoder-decoder
    model.add_loss(
        losses=inverse_reg(inpt, encoder, decoder, lambda_=lambda_),
        inputs=[inpt]
    )

    return model, encoder, dynamics, decoder




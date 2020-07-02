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

from src.regularizers import *
from src.common import *
from src.autoencoders import *


def lyapunov_autoencoder(snapshot_shape, output_dims, lambda_, kappa, gamma,
        no_stability=False, sizes=(40,25,15), all_models=False):
    """
    Create a lyapunov autoencoder model
    Args:
        snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
        output_dims (int): number of output channels
        lambda_ (float): weighting factor for inverse regularizer
        kappa (float): weighting factor for stability regularizer
        sizes (tuple of int): depth of layers in decreasing order of size. default (40,25,15)
    Returns:
        a Keras Model of the autoencoder
    """
    inpt = Input(snapshot_shape)
    print("Autoencoder Input shape:", inpt.shape)

    large, medium, small = sizes

    encoder = AutoencoderBlock((large, medium, small), name="encoder", 
        batchnorm_last=True)
    dynamics = LyapunovStableDense(kappa=kappa, gamma=gamma, no_stab=no_stability,
        units=small, name="lyapunovstable-dynamics")
    decoder = AutoencoderBlock((medium, large, output_dims), name="decoder")

    x = encoder(inpt)
    x = dynamics(x)
    x = decoder(x)

    model = Model(inpt, x)

    # inverse regularizer of encoder-decoder
    inv_loss = lambda_ * inverse_reg(inpt, encoder, decoder)

    model.add_loss(inv_loss)
    model.add_metric(inv_loss, name="inverse_loss", aggregation='mean')

    if all_models:
        return model, encoder, dynamics, decoder
    return model





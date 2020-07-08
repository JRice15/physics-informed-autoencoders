import math

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.models import Model
from keras import Input
from keras.layers import Layer, Dense


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




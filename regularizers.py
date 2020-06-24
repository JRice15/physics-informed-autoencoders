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


def inverse_reg(x, encoder: Model, decoder: Model, lambda_):
    """
    regularizer to enforce that the decoder is the inverse of the encoder. 
    Equation 9 per Erichson et al.  
    Args:
        encoder, decoder: Keras Models
        lambda_: weighting hyperparameter
    """
    q = encoder.predict(x)
    q_pred = encoder.predict( decoder.predict(q) )
    norm = K.sum(K.square(q - q_pred))
    return lambda_ * norm


def lyapunov_stability_reg(x, dynamics: Model, kappa):
    """
    regularizes stability via eigenvalues of P matrix from equation
    """
    ...





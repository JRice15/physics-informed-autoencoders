import math

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros, GlorotNormal
from keras.models import Model
from keras import Input
from keras.layers import Layer, Dense


class SymmetricGlorotNormal(GlorotNormal):

    def __call__(self, shape, dtype=None):
        kernel = super().__call__(shape, dtype).numpy()
        assert len(shape) == 2 and shape[0] == shape[1]
        for r in range(int(shape[0])):
            for c in range(int(shape[1])):
                if c < r:
                    kernel[r][c] = kernel[c][r]
        return kernel


class LyapunovStableDense(Dense):
    """
    Dense layer with optional Lyapunov stability regularization
    Args:
        kappa, gamma: regularization hyperparams
        no_stab (bool): whether to have no stability regularization
    """

    def __init__(self, kappa, gamma=4, no_stab=False, *args, **kwargs):
        if not no_stab:
            kwargs["kernel_regularizer"] = self.lyapunov_stability_reg()
        super().__init__(*args, 
            kernel_initializer=SymmetricGlorotNormal(),
            bias_initializer=zeros(),
            **kwargs)
        self.kappa = kappa
        self.gamma = gamma

    def lyapunov_stability_reg(self):
        """
        regularizes stability via eigenvalues of P matrix from Equation 21
        Args:
            dynamics: ComposedLayers
            layer_size: number n that is the size of the n*n weights in the dynamics layer
            kappa: weighting hyperparameter for this loss term
            gamma: hyperparameter, divisor in loss term exponent, default 4
        """
        def stability_regularizer(omega):
            """
            regularizer that accepts weight kernel and returns loss
            """
            # solve discrete lyapunov equation for P
            omegaT = tf.transpose(omega)
            omegaT = tf.linalg.LinearOperatorFullMatrix(omegaT)
            kron = tf.linalg.LinearOperatorKronecker([omegaT, omegaT], is_square=True)
            kron = kron.add_to_tensor( -tf.eye(kron.shape[-1]) )
            pseudinv = tf.linalg.pinv(kron, validate_args=True) # tf.linalg.pinv requires tf version != 2.0
            neg_Ivec = vec( -tf.eye(self.units) )
            Pvec = tf.linalg.matvec(pseudinv, neg_Ivec)
            P = unvec(Pvec, self.units)

            tf.print("nan:", tf.math.reduce_any(tf.math.is_nan(P)))
            tf.print("inf:", tf.math.reduce_any(tf.math.is_inf(P)))

            # calculate eigenvalues of P
            eigvalues, eigvectors = tf.linalg.eigh(P)
            # eigvalues = tf.cast(eigvalues, tf.float32)

            # calculate loss
            prior = tf.exp((eigvalues - 1) / self.gamma)
            prior = tf.where(eigvalues < 0, prior, tf.zeros(prior.shape))
            loss = self.kappa * tf.reduce_sum(prior)
            
            self.add_metric(value=loss, name="stability", aggregation='mean')

            return loss
        
        return stability_regularizer




def inverse_reg(x, encoder, decoder):
    """
    regularizer to enforce that the decoder is the inverse of the encoder. 
    Equation 9 per Erichson et al.  
    Args:
        encoder, decoder: ComposedLayers
        lambda_: weighting hyperparameter for this loss term
    """
    q = encoder(x)
    q_pred = encoder(decoder(q))
    norm = tf.reduce_mean(tf.square(q - q_pred))
    return norm


def unvec(x, n):
    """
    convert vector back to n*n square matrix, using the vector as n stacked columns
    """
    x = tf.split(x, n)
    x = tf.convert_to_tensor(x)
    return tf.transpose(x)

def vec(X):
    """
    stacking columns to create a vector from matrix X
    """
    x = tf.unstack(X, axis=-1)
    x = tf.concat(x, axis=0)
    return x





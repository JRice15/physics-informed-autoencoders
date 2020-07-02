import math

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.models import Model
from keras import Input
from keras.layers import Layer, Dense



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
            kernel_initializer=glorot_normal(),
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
            # tf.print("\nomega nan:", tf.math.reduce_any(tf.math.is_nan(omega)),
            #     "omega inf:", tf.math.reduce_any(tf.math.is_inf(omega)))
            # tf.print("mean:", K.mean(omega), "max:", K.max(omega), "min:", K.min(omega))

            # solve discrete lyapunov equation for P
            omegaT = tf.transpose(omega)
            omegaT = tf.linalg.LinearOperatorFullMatrix(omegaT)
            kron = tf.linalg.LinearOperatorKronecker([omegaT, omegaT], is_square=True)
            kron = kron.add_to_tensor( -tf.eye(kron.shape[-1]) )
            pseudinv = tf.linalg.pinv(kron, validate_args=True) # tf.linalg.pinv requires tf version != 2.0
            neg_Ivec = vec( -tf.eye(self.units) )
            Pvec = tf.linalg.matvec(pseudinv, neg_Ivec)
            P = unvec(Pvec, self.units)

            # tf.print("kron nan:", tf.math.reduce_any(tf.math.is_nan(kron)),
            #     "pseudinv nan:", tf.math.reduce_any(tf.math.is_nan(pseudinv)),
            #     "negIvec nan:", tf.math.reduce_any(tf.math.is_nan(neg_Ivec)),
            #     "Pvec nan:", tf.math.reduce_any(tf.math.is_nan(Pvec)),
            #     "P nan:", tf.math.reduce_any(tf.math.is_nan(P)))

            # calculate eigenvalues of P
            eigvalues, eigvectors = tf.linalg.eigh(P)
            # eigvalues = tf.cast(eigvalues, tf.float32)

            # calculate loss
            neg_eigvalues = tf.gather(eigvalues, tf.where(eigvalues < 0))
            prior = tf.exp((neg_eigvalues - 1) / self.gamma)
            loss = self.kappa * tf.reduce_sum(prior)
            
            # tf.print("prior nan:", tf.math.reduce_any(tf.math.is_nan(prior)),
            #     "loss nan:", tf.math.reduce_any(tf.math.is_nan(loss)))
            # tf.print("loss:", loss)

            self.add_metric(value=loss, name="stability_loss", aggregation='mean')

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
    # q = encoder(x)
    # q_pred = encoder(decoder(q))
    # norm = tf.reduce_mean(tf.square(q - q_pred))
    x_pred = decoder(encoder(x))
    norm = tf.reduce_mean(tf.square(x - x_pred))
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







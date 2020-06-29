import math

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.models import Model
from keras.layers import Layer, Dense


class LyapunovStableDense(Dense):

    def __init__(self, kappa, gamma=4, *args, **kwargs):
        super().__init__(*args, 
            kernel_initializer=glorot_normal(),
            bias_initializer=zeros(),
            kernel_regularizer=get_lyapunov_stability_regularizer(kwargs["units"], kappa, gamma),
            **kwargs)
        self.kappa = kappa
        self.gamma = gamma

    def __call__(self, inputs):
        outputs = super().__call__(inputs)
        # stability_loss = self.lyapunov_stability_reg()
        # self.add_loss(stability_loss)
        # self.add_metric(stability_loss, "stability_loss")
        return outputs

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
            loss = self.kappa * K.mean(K.square(omega))
            self.add_metric(value=loss, name="testmet", aggregation="mean")
            return loss

            # solve discrete lyapunov equation for P
            omegaT = tf.transpose(omega)
            omegaT = tf.linalg.LinearOperatorFullMatrix(omegaT)
            kron = tf.linalg.LinearOperatorKronecker([omegaT, omegaT])
            kron = kron.add_to_tensor( -tf.eye(kron.shape[-1]) )
            pseudinv = tf.linalg.pinv(kron) # tf.linalg.pinv requires tf version != 2.0
            Ivec = vec(tf.eye(tf.sqrt(tf.cast(pseudinv.shape[-1], tf.float32))))
            Pvec = tf.linalg.matvec(pseudinv, Ivec)
            P = unvec(Pvec, self.units)

            # calculate eigenvalues of P
            eigvalues, eigvectors = tf.linalg.eigh(P)

            # calculate loss
            prior = tf.exp((eigvalues - 1) / self.gamma)
            prior = tf.where(eigvalues < 0, prior, tf.zeros(prior.shape))
            loss = tf.reduce_sum(prior)
            
            return self.kappa * loss
        
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


def get_lyapunov_stability_regularizer(size, kappa, gamma=4):
    """
    regularizes stability via eigenvalues of P matrix from Equation 21
    Args:
        layer: Layer
        size: number n that is the size of the n*n weights in the dynamics layer
        kappa: weighting hyperparameter for this loss term
        gamma: hyperparameter, divisor in loss term exponent, default 4
    """
    def stability_regularizer(omega):
        return kappa * K.mean(K.square(omega))

        # solve discrete lyapunov equation for P
        omegaT = tf.transpose(omega)
        omegaT = tf.linalg.LinearOperatorFullMatrix(omegaT)
        kron = tf.linalg.LinearOperatorKronecker([omegaT, omegaT])
        kron = kron.add_to_tensor( -tf.eye(kron.shape[-1]) )
        pseudinv = tf.linalg.pinv(kron) # tf.linalg.pinv requires tf version != 2.0
        Ivec = vec(tf.eye(tf.sqrt(tf.cast(pseudinv.shape[-1], tf.float32))))
        Pvec = tf.linalg.matvec(pseudinv, Ivec)
        P = unvec(Pvec, size)

        # calculate eigenvalues of P
        eigvalues, eigvectors = tf.linalg.eigh(P)

        # calculate loss
        prior = tf.exp((eigvalues - 1) / gamma)
        prior = tf.where(eigvalues < 0, prior, tf.zeros(prior.shape))
        loss = tf.reduce_sum(prior)
        
        return kappa * loss
    
    return stability_regularizer




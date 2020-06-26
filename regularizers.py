import math

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.activations import tanh
from keras.initializers import glorot_normal, zeros
from keras.models import Model

import jax
import scipy


def inverse_reg(x, encoder, decoder, lambda_):
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
    loss = lambda_ * norm
    return loss



@tf.custom_gradient
def solve_lyapunov(A):
    """
    use Jax to compute gradient of scipy solution to discrete lyapunov equation
    https://gist.github.com/shoyer/5f72853c2788e99e785f4737ee8a6ae1#file-jax_tf_eager_autodiff_compatibility-ipynb
    """
    print("solve_lyapunov A shape:", A.shape)
    # conversions
    to_jax = lambda x: jax.device_put(jax.numpy.asarray(np.asarray(x)))
    to_tf = lambda x: tf.convert_to_tensor(np.asarray(x))

    def lyapunov_fn(x):
        return solve_discrete_lyapunov_bilinear(
            a=x, 
            q=np.eye(A.shape[-1]),
        )
    P, grad_fn = jax.vjp(lyapunov_fn, A.numpy())

    def tf_grad_fn(d):
        out, = to_tf(grad_fn(to_jax(d)))
        return out

    return to_tf(P), tf_grad_fn
    


def lyapunov_stability_reg(dynamics, kappa, gamma=4):
    """
    regularizes stability via eigenvalues of P matrix from Equation 19, and
    Reference 37 (Fausset & Fulton)
    Args:
        dynamics: ComposedLayers
        kappa: weighting hyperparameter for this loss term
        gamma: hyperparameter, divisor in loss term exponent, default 4
    """

    from modified_jax_linalg import solve_discrete_lyapunov_bilinear

    # first and only Dense layer of dynamics
    omega = dynamics.layers[0]
    # get weight matrix (bias is index 1)
    omega = omega.weights[0]

    with tf.GradientTape() as tape:
        tape.watch(omega)

        # solve lyapunov equation for P
        omegaT = tf.transpose(omega)
        P = solve_lyapunov(omegaT)

    # calculate eigenvalues of P
    eigvalues, eigvectors = tf.linalg.eigh(P)

    # calculate loss
    prior = tf.exp((eigvalues - 1) / 4)
    prior = tf.where(eigvalues < 0, prior, tf.zeros(prior.shape))
    loss = kappa * tf.reduce_sum(prior)

    # grads = tape.gradient(loss, [omega])
    # optimizer.apply_gradients(zip(loss, [omega]))
    
    return loss


##########
# Method 2
##########

def unvec(x, n):
    """
    convert vector to n*n square matrix, using the vector as n stacked columns
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


def lyapunov_stability_reg_2(dynamics, layer_size, kappa, gamme=4):
    """
    regularizes stability via eigenvalues of P matrix from Equation 19, and
    Reference 37 (Fausset & Fulton)
    Args:
        dynamics: ComposedLayers
        layer_size: number n that is the size of the n*n weights in the dynamics layer
        kappa: weighting hyperparameter for this loss term
        gamma: hyperparameter, divisor in loss term exponent, default 4
    """
    # first and only Dense layer of dynamics
    omega = dynamics.layers[0]
    # get weight matrix (bias is index 1)
    omega = omega.weights[0]

    # solve lyapunov equation for P
    omegaT = tf.transpose(omega)
    print("Omega shape:", omegaT.shape)
    omegaT = tf.linalg.LinearOperatorFullMatrix(omegaT)
    kron = tf.linalg.LinearOperatorKronecker([omegaT, omegaT])
    kron = kron.add_to_tensor( -tf.eye(kron.shape[-1]) )
    # requires tf != 2.0
    pseudinv = tf.linalg.pinv(kron)
    Ivec = vec(tf.eye(tf.sqrt(tf.cast(pseudinv.shape[-1], tf.float32))))
    print("Ivec shape:", Ivec.shape)
    Pvec = tf.linalg.matvec(pseudinv, Ivec)
    print("Pvec shape:", Pvec.shape)
    P = unvec(Pvec, layer_size)
    print("P shape:", P.shape)

    # calculate eigenvalues of P
    eigvalues, eigvectors = tf.linalg.eigh(P)

    # calculate loss
    prior = tf.exp((eigvalues - 1) / 4)
    prior = tf.where(eigvalues < 0, prior, tf.zeros(prior.shape))
    loss = kappa * tf.reduce_sum(prior)
    
    return loss

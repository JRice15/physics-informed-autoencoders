import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.initializers import glorot_normal, zeros
from keras.models import Model
from keras.activations import tanh
import jax
import scipy


def inverse_reg(x, encoder, decoder, lambda_):
    """
    regularizer to enforce that the decoder is the inverse of the encoder. 
    Equation 9 per Erichson et al.  
    Args:
        encoder, decoder: composed layers
        lambda_: weighting hyperparameter for this loss term
    """
    q = encoder(x)
    q_pred = encoder(decoder(q))
    norm = K.sum(K.square(q - q_pred))
    return lambda_ * norm



@tf.custom_gradient
def solve_lyapunov(A):
    """
    use Jax to compute gradient of scipy solution to discrete lyapunov equation
    https://gist.github.com/shoyer/5f72853c2788e99e785f4737ee8a6ae1#file-jax_tf_eager_autodiff_compatibility-ipynb
    """
    print("solve_lyapunov A shape:", A.shape)
    # conversions
    to_jax = lambda x: jax.numpy.asarray(np.asarray(x))
    to_tf = lambda x: tf.convert_to_tensor(np.asarray(x))

    def lyapunov_fn(x): 
        return scipy.linalg.solve_discrete_lyapunov(
            a=x, 
            q=to_jax(np.eye(A.shape[-1])),
        )
    P, grad_fn = jax.vjp(lyapunov_fn, to_jax(A))

    def tf_grad_fn(d):
        out, = to_tf(grad_fn(to_jax(d)))
        return out

    return to_tf(P), tf_grad_fn
    


def lyapunov_stability_reg(optimizer, dynamics: Model, kappa, gamma=4):
    """
    regularizes stability via eigenvalues of P matrix from Equation 19, and
    Reference 37 (Fausset & Fulton)
    Args:
        dynamics: Keras Model
        kappa: weighting hyperparameter for this loss term
        gamma: hyperparameter, divisor in loss term exponent, default 4
    """
    # first and only Dense layer of dynamics
    omega = dynamics.layers[0]
    # get weight matrix
    omega = omega.get_weights()[0]

    with tf.GradientTape() as tape:
        tape.watch(omega)

        # solve lyapunov equation for P
        omegaT = omega.T
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





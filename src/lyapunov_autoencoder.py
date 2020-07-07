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


class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.01
    wd = 1e-6
    epochs = 6000
    batchsize = 34
    lambda_ = 3 # inverse regularizer weight
    kappa = 3 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes = (40, 25, 15) # largest to smallest


def make_run_name(args):
    run_name = args.name + ".lyapunov."
    if args.no_stability:
        run_name += "nostability."
    else:
        run_name += "stability."
    run_name += "l{}_k{}_g{}.".format(args.lambd, args.kappa, args.gamma)
    run_name += "{}epochs_{}batchsize_{}lr_{}wd.".format(args.epochs, args.batchsize, args.lr, args.wd)
    run_name += "s{}_{}_{}.".format(*args.sizes)
    return run_name


def vec(X):
    """
    stacking columns to create a vector from matrix X
    """
    x = tf.unstack(X, axis=-1)
    x = tf.concat(x, axis=0)
    return x


def unvec(x, n):
    """
    convert vector back to n*n square matrix, using the vector as n stacked columns
    """
    x = tf.split(x, n)
    x = tf.convert_to_tensor(x)
    return tf.transpose(x)


class LyapunovStableDense(Dense):
    """
    Dense layer with optional Lyapunov stability regularization
    Args:
        kappa, gamma: regularization hyperparams
        no_stab (bool): whether to have no stability regularization
    """

    def __init__(self, kappa, weight_decay, gamma=4, no_stab=False, *args, **kwargs):
        self.weight_decay = weight_decay
        self.kappa = kappa
        self.gamma = gamma
        if not no_stab:
            kwargs["kernel_regularizer"] = self.lyapunov_stability_reg()
        super().__init__(*args, 
            kernel_initializer=glorot_normal(),
            bias_initializer=zeros(),
            **kwargs)


    def lyapunov_stability_reg(self):
        """
        regularizes stability via eigenvalues of P matrix from Equation 21
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

            # calculate eigenvalues of P
            eigvalues, eigvectors = tf.linalg.eigh(P)
            # eigvalues = tf.cast(eigvalues, tf.float32)

            # calculate loss
            neg_eigvalues = tf.gather(eigvalues, tf.where(eigvalues < 0))
            prior = tf.exp((neg_eigvalues - 1) / self.gamma)
            loss = self.kappa * tf.reduce_sum(prior)

            self.add_metric(value=loss, name="stability_loss", aggregation='mean')

            # add weight decay
            loss += self.weight_decay * tf.reduce_sum(omega ** 2)

            return loss
        
        return stability_regularizer



def lyapunov_autoencoder(snapshot_shape, output_dims, lambda_, kappa, gamma,
        no_stability, sizes, weight_decay):
    """
    Create a lyapunov autoencoder model
    Args:
        snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
        output_dims (int): number of output channels
        lambda_ (float): weighting factor for inverse regularizer
        kappa (float): weighting factor for stability regularizer
        sizes (tuple of int): depth of layers in decreasing order of size. default (40,25,15)
    Returns:
        4 Keras Model of the autoencoder
    """
    inpt = Input(snapshot_shape)
    print("Autoencoder Input shape:", inpt.shape)

    large, medium, small = sizes

    encoder = AutoencoderBlock((large, medium, small), weight_decay, name="encoder", 
        batchnorm_last=True)
    dynamics = LyapunovStableDense(kappa=kappa, gamma=gamma, weight_decay=weight_decay,
        no_stab=no_stability, units=small, name="lyapunovstable-dynamics")
    decoder = AutoencoderBlock((medium, large, output_dims), weight_decay,
        name="decoder")

    x = encoder(inpt)
    x = dynamics(x)
    x = decoder(x)

    model = Model(inpt, x)

    # inverse regularizer of encoder-decoder
    inv_loss = lambda_ * inverse_reg(inpt, encoder, decoder)

    model.add_loss(inv_loss)
    model.add_metric(inv_loss, name="inverse_loss", aggregation='mean')

    return model, encoder, dynamics, decoder







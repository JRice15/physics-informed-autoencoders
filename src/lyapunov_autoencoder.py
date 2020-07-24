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
from keras import metrics, losses

from src.common import *
from src.dense_autoencoders import *
from src.conv_autoencoders import *
from src.output_results import *


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


class LyapunovStableLayer(Layer):
    """
    Dense layer with optional Lyapunov stability regularization
    Args:
        kappa, gamma: regularization hyperparams
        no_stab (bool): whether to have no stability regularization
    """

    def __init__(self, units, kappa, weight_decay, name, gamma=4, no_stab=False, **kwargs):
        super().__init__(name=name)

        self.units = units
        self.weight_decay = weight_decay
        self.kappa = kappa
        self.gamma = gamma
        self.no_stab = no_stab
        if not no_stab:
            kwargs["kernel_regularizer"] = self.lyapunov_stability_reg()
        kwargs["kernel_initializer"] = glorot_normal()
        kwargs["bias_initializer"] = zeros()

        self.dyn_layer = Dense(units=units, name=name+"dense", **kwargs)


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

    def call(self, x):
        return self.dyn_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "kappa": self.kappa,
            "gamma": self.gamma,
            "weight_decay": self.weight_decay,
            "no_stab": self.no_stab,
        })
        return config


class LyapunovAutoencoder(BaseAE):

    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)
        self.build_enc_dec(args, dataset.input_shape[-1])
        self.build_model(
            snapshot_shape=dataset.input_shape,
            output_dims=dataset.input_shape[-1],
            kappa=args.kappa,
            lambda_=args.lambd,
            gamma=args.gamma,
            no_stability=args.no_stability,
            weight_decay=args.wd,
        )

    def build_enc_dec(self, args, output_dims):
        if args.convolutional:
            raise NotImplementedError("convolutional lyapunov not implemented yet")
        else:
            large, medium, small = args.sizes
            self.encoder = DenseAutoencoderBlock(args.activation, (large, medium, small), args.wd, name="encoder", 
                batchnorm_last=True)
            self.decoder = DenseAutoencoderBlock(args.activation, (medium, large, output_dims), args.wd,
                name="decoder")

    def build_model(self, snapshot_shape, output_dims, lambda_, kappa, gamma,
            no_stability, weight_decay):
        """
        Create a lyapunov autoencoder model
        Args:
            snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
            output_dims (int): number of output channels
            lambda_ (float): weighting factor for inverse regularizer
            kappa (float): weighting factor for stability regularizer
            sizes (tuple of int): depth of layers in decreasing order of size. default (40,25,15)
        """
        inpt = Input(snapshot_shape)
        print("Autoencoder Input shape:", inpt.shape)

        x = self.encoder(inpt)
        dyn_units = x.shape[-1]
        self.dynamics = LyapunovStableLayer(kappa=kappa, gamma=gamma, weight_decay=weight_decay,
            no_stab=no_stability, units=dyn_units, name="lyapunov-dynamics")

        x = self.dynamics(x)
        x = self.decoder(x)

        self.model = Model(inpt, x)

        # inverse regularizer of encoder-decoder
        inv_loss = lambda_ * inverse_reg(inpt, self.encoder, self.decoder)

        self.model.add_loss(inv_loss)
        self.model.add_metric(inv_loss, name="inverse_loss", aggregation='mean')

    
    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=losses.mse, 
            metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])

    def make_run_name(self):
        args = self.args
        run_name = args.name + ".lyap."
        if args.convolutional:
            run_name += "conv."
        if args.no_stability:
            run_name += "nostabl."
        else:
            run_name += "stabl."
        run_name += "l{}_k{}_g{}.".format(args.lambd, args.kappa, args.gamma)
        run_name += self.run_name_common_suffix()
        return run_name

    def format_data(self):
        # targets one timestep ahead of inputs
        self.Y = self.dataset.X[:-1]
        self.X = self.dataset.X[1:]
        Ytest = self.dataset.Xtest[:-1]
        Xtest = self.dataset.Xtest[1:]
        self.val_data = (Xtest, Ytest)

    def train(self, callbacks):
        self.format_data()
        H = self.model.fit(
            x=self.X, 
            y=self.Y,
            batch_size=self.args.batchsize,
            epochs=self.args.epochs,
            callbacks=callbacks,
            validation_data=self.val_data,
            verbose=2,
        )
        return H

    def get_pipeline(self):
        """
        get forward prediction pipeline
        """
        return (self.encoder, self.dynamics, self.decoder)

    def save_eigenvals(self):
        output_eigvals(self.dynamics.weights[0], self.make_run_name())


CUSTOM_OBJ_DICT.update({
    "LyapunovStableLayer": LyapunovStableLayer
})

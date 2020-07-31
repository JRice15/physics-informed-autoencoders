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



class ConstantLayer(Layer):
    """
    guess a single constant value
    """

    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.w = self.add_weight(
            shape=None,
            initializer=lambda *args, **kwargs: 0.0,
            trainable=True
        )
    
    def build(self, input_shape):
        self.var_shape = input_shape[1:]
        assert len(self.var_shape) < 3
    
    def call(self, x=None):
        tf.print("guess: " + str(self.w.numpy()))
        w = K.reshape(self.w, (1,1))
        w = K.repeat_elements(w, self.var_shape[0], 0)
        if len(self.var_shape > 1):
            w = K.repeat_elements(w, self.var_shape[1], 1)
        return w



class NaiveBaseline(BaseAE):

    def __init__(self, kind, args, dataset=None):
        super().__init__(args, dataset)
        self.build_enc_dec()
        self.build_model(
            kind=kind,
            snapshot_shape=dataset.input_shape,
        )

    def build_enc_dec(self):
        # identity layers
        self.encoder = Layer()
        self.decoder = Layer()

    def build_model(self, kind, snapshot_shape):
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

        if kind in ("const", "constant"):
            self.guess = ConstantLayer("const")
        elif kind == "identity":
            self.guess = Layer()
        else:
            raise ValueError("Unknown model type '{}'".format(kind))
        x = self.guess(inpt)

        self.model = Model(inpt, x)
    
    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=losses.mse, 
            metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])

    def make_run_name(self):
        args = self.args
        run_name = args.name + ".const."
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
        raise NotImplementedError()

    def save_eigenvals(self):
        return self.encoder, self.guess, self.decoder


CUSTOM_OBJ_DICT.update({
    "NaiveBaseline": NaiveBaseline,
    "ConstantLayer": ConstantLayer,
})

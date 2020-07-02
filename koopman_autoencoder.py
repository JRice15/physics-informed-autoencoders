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

from regularizers import *
from common import *


class BackwardDynamicsInitializer():

    def __init__(self, forward_weights):
        self.forward_weights = forward_weights
    
    def __call__(self, shape, dtype=None):
        assert self.forward_weights.shape == shape
        return tf.linalg.pinv(tf.transpose(self.forward_weights))


def koopman_autoencoder(snapshot_shape, output_dims, pred_steps=5,
        sizes=(40,15)):
    """
    Create a Koopman autoencoder
    Args:
        snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
        output_dims (int): number of output channels
        kappa (float): bottleneck
        sizes (tuple of int): depth of layers in decreasing order of size. default
    Returns:
        full: Keras Model for full autoencoder
        encoder: ComposedLayers
        dynamics: Layer
        decoder: ComposedLayers
    """
    # add channels
    total_inpt_snapshots = (pred_steps * 2) + 1
    inpt = Input((total_inpt_snapshots,) + snapshot_shape)
    print("Autoencoder Input shape:", inpt.shape)

    intermediate = sizes[0]
    bottleneck = sizes[1]

    #--------------------------- Encoder --------------------------------------
    encoder_layers = make_fc_block(intermediate, name="enc1")
    encoder_layers += make_fc_block(intermediate, name="enc2")
    encoder_layers += make_fc_block(bottleneck, activate=False, name="enc3")

    encoder = ComposedLayers(encoder_layers)

    #--------------------------- Forward Dynamics -----------------------------
    forward = Dense(bottleneck, use_bias=False, 
        kernel_initializer=glorot_normal())

    #--------------------------- Backward Dynamics ----------------------------
    backward = Dense(bottleneck, use_bias=False, 
        kernel_initializer=BackwardDynamicsInitializer(forward.weights[0]))

    #--------------------------- Decoder --------------------------------------
    decoder_layers = make_fc_block(intermediate, name="dec1")
    decoder_layers += make_fc_block(intermediate, name="dec2")
    decoder_layers += make_fc_block(output_dims, activate=False, name="dec3")

    decoder = ComposedLayers(decoder_layers)

    #--------------------------- Create Model ---------------------------------

    encoded_out = encoder(inpt)
    forward_out = []
    backward_out = []
    # for each step size
    for s in range(1, pred_steps+1):
        # compute the forward and backward predictions
        x_f = encoded_out
        x_b = encoded_out
        for _ in range(s):
            x_f = forward(x_f)
            x_b = backward(x_b)
        forward_out.append(decoder(x_f))
        backward_out.append(decoder(x_b))
    
    model = Model(inputs=inpt, outputs=[forward_out, backward_out])

    model.add_loss()

    return model



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
        a Keras Model. This model will accept one input of 
            (x images before, input image, x images after), where x is pred_steps
    """
    total_inpt_snapshots = (pred_steps * 2) + 1
    inpt = Input((total_inpt_snapshots,) + snapshot_shape)
    print("Autoencoder Input shape:", inpt.shape)

    intermediate = sizes[0]
    bottleneck = sizes[1]

    encoder = AutoencoderBlock((intermediate, intermediate, bottleneck),
        name="encoder")
    forward = Dense(bottleneck, use_bias=False, name="forward-dynamics",
        kernel_initializer=glorot_normal())
    backward = Dense(bottleneck, use_bias=False, name="backward-dynamics",
        kernel_initializer=BackwardDynamicsInitializer(forward.weights[0]))
    decoder = AutoencoderBlock((intermediate, intermediate, output_dims),
        name="decoder")

    # predict pred_steps into the future
    current_image = inpt[pred_steps]
    encoded_out = encoder(current_image)
    # outputs is list [earliest_backward_pred, ... latest_forward_pred]
    outputs = []
    # for each step size
    for s in range(1, pred_steps+1):
        # compute the forward and backward predictions
        x_f = encoded_out
        x_b = encoded_out
        for _ in range(s):
            x_f = forward(x_f)
            x_b = backward(x_b)
        outputs.append(decoder(x_f))
        outputs.insert(0, decoder(x_b))
    
    model = Model(inputs=inpt, outputs=outputs)

    true = tf.concat([inpt[:pred_steps], inpt[pred_steps+1:]], axis=0)
    loss = tf.reduce_mean((true - outputs) ** 2)
    model.add_loss(loss, name="mse", aggregation="mean")

    return model



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
    epochs = 6000
    batchsize = 34
    pred_steps = 8
    forward = 1
    backward = 0.1
    identity = 1
    consistency = 0.01
    sizes = (2*16, 10) # largest to smallest



def make_run_name(args):
    run_name = args.name + ".koopman."
    run_name += "{}steps.".format(args.pred_steps)
    run_name += "f{}_b{}_i{}_c{}.".format(args.forward, args.backward, args.identity, args.consistency)
    run_name += "{}epochs_{}batchsize_{}lr.".format(args.epochs, args.batchsize, args.lr)
    run_name += "s{}_{}.".format(*args.sizes)
    return run_name


class InverseDynamicsInitializer():

    def __init__(self, pair_layer):
        self.pair_weights = pair_layer.weights[0]
    
    def __call__(self, shape, dtype=None):
        assert self.pair_weights.shape == shape
        return tf.linalg.pinv(tf.transpose(self.pair_weights))


class KoopmanConsistencyLayer(Dense):
    """
    enforce consistency of forward-backward predictions
    """

    def __init__(self, pair_layer, cons_wt, units, name, use_bias=False):
        self.pair_layer = pair_layer
        self.cons_wt = cons_wt
        super().__init__(
            units=units, 
            name=name, 
            use_bias=use_bias,
            kernel_initializer=InverseDynamicsInitializer(pair_layer),
            kernel_regularizer=self.consistency_reg()
        )
    
    def consistency_reg(self):
        """
        regularize consistency of forward-backward dynamics
        """
        def regularizer(D):
            """
            D: backward weights
            """
            C = self.pair_layer.weights[0]

            loss = 0
            # units == kappa in paper
            for k in range(1, self.units+1):
                I = tf.eye(k)
                loss += tf.reduce_sum(
                    (tf.matmul(D[:k,:], C[:,:k]) - I) ** 2
                ) / (2*k)
                loss += tf.reduce_sum(
                    (tf.matmul(C[:k,:], D[:,:k]) - I) ** 2
                ) / (2*k)
            loss = self.cons_wt * loss

            self.add_metric(loss, name="cons_loss", aggregation="mean")
            return loss
        
        return regularizer
        


def koopman_autoencoder(snapshot_shape, output_dims, fwd_wt, bwd_wt, id_wt, 
        cons_wt, pred_steps, sizes, all_models=False):
    """
    Create a Koopman autoencoder
    Args:
        snapshot_shape (tuple of int): shape of snapshots, without batchsize or channels
        output_dims (int): number of output channels
        fwd_wt, bwd_wt, id_wt, cons_wt: forward, backward, identity, and consistency regularizer weights
        sizes (tuple of int): depth of layers in decreasing order of size. default
    Returns:
        a Keras Model. This model will accept one input of 
            (x images before, input image, x images after), where x is pred_steps
    """
    total_inpt_snapshots = (pred_steps * 2) + 1
    inpt = Input((total_inpt_snapshots,) + snapshot_shape)
    tf.print("inpt shape:", inpt.shape)
    current = inpt[:,pred_steps,:]
    tf.print("current shape:", current.shape)
    print("Autoencoder Input shape:", inpt.shape, "current shape:", current.shape)

    intermediate, bottleneck = sizes

    encoder = AutoencoderBlock((intermediate, intermediate, bottleneck),
        name="encoder")
    forward = Dense(bottleneck, use_bias=False, name="forward-dynamics",
        kernel_initializer=glorot_normal())
    # needed to build fwd layer
    _ = forward(encoder(current))
    backward = KoopmanConsistencyLayer(pair_layer=forward, cons_wt=cons_wt, 
        units=bottleneck, use_bias=False, name="backward-dynamics")
    decoder = AutoencoderBlock((intermediate, intermediate, output_dims),
        name="decoder")

    # predict pred_steps into the future
    encoded_out = encoder(current)
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

    # Forward and Backward Dynamics Regularizers
    bwd_pred = tf.stack(outputs[:pred_steps], axis=1)
    bwd_true = inpt[:,:pred_steps,:]
    fwd_pred = tf.stack(outputs[pred_steps:], axis=1)
    fwd_true = inpt[:,pred_steps+1:,:]
    print("pred shape:", fwd_pred.shape, bwd_pred.shape)
    print("true shape:", fwd_true.shape, bwd_true.shape)
    tf.print(fwd_true.shape)
    bwd_loss = bwd_wt * tf.reduce_mean((bwd_true - bwd_pred) ** 2)
    fwd_loss = fwd_wt * tf.reduce_mean((fwd_true - fwd_pred) ** 2)
    model.add_loss(bwd_loss)
    model.add_loss(fwd_loss)
    model.add_metric(bwd_loss, name="bwd_loss", aggregation="mean")
    model.add_metric(fwd_loss, name="fwd_loss", aggregation="mean")

    # Encoder-Decoder Identity
    print("current shape:", current.shape)
    id_loss = id_wt * inverse_reg(current, encoder, decoder)
    model.add_loss(id_loss)
    model.add_metric(id_loss, name="id_loss", aggregation="mean")

    return model, encoder, forward, backward, decoder



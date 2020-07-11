import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.initializers import glorot_normal, zeros, Initializer
from keras.layers import (Activation, Add, BatchNormalization, Concatenate,
                          Conv2D, Cropping2D, Dense, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D,
                          Lambda, Layer, LeakyReLU, MaxPooling2D, ReLU,
                          Reshape, Softmax, Subtract, UpSampling2D,
                          ZeroPadding2D, add)
from keras.models import Model
from keras.activations import tanh
from keras import regularizers

from src.common import *
from src.autoencoders import *
from src.output_results import *


class InverseDynamicsInitializer(Initializer):
    """
    initialize a layers weights to be the (pseudo-)inverse of another layer's
    """

    def __init__(self, in_shape, pair_layer):
        self.in_shape = in_shape
        # build pair layer
        _ = pair_layer(Input(in_shape))
        self.pair_layer = pair_layer
    
    def __call__(self, shape, dtype=None):
        pair_weights = self.pair_layer.weights[0]
        assert pair_weights.shape == shape
        return tf.linalg.pinv(tf.transpose(pair_weights))

    def get_config(self):
        return {
            "pair_layer": self.pair_layer,
            "in_shape": self.in_shape
        }


class KoopmanConsistencyLayer(Dense):
    """
    enforce consistency of forward-backward predictions
    """

    def __init__(self, in_shape, pair_layer, cons_wt, units, name, weight_decay, 
            use_bias=False, *args, **kwargs):
        self.in_shape = in_shape
        self.pair_layer = pair_layer
        self.cons_wt = cons_wt
        self.weight_decay = weight_decay
        kwargs["kernel_initializer"] = InverseDynamicsInitializer(in_shape, pair_layer)
        kwargs["kernel_regularizer"] = self.consistency_reg()
        kwargs["bias_initializer"] = zeros()
        super().__init__(
            *args, **kwargs,
            units=units, 
            name=name, 
            use_bias=use_bias,
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

            # add weight decay
            loss += self.weight_decay * tf.reduce_sum(D ** 2)

            return loss
        
        return regularizer
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "pair_layer": self.pair_layer,
            "cons_wt": self.cons_wt,
            "weight_decay": self.weight_decay,
            "in_shape": self.in_shape
        })
        return config


class KoopmanAutoencoder(BaseAE):

    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)
        if args.bwd_steps > 0:
            self.has_bwd = True
        else:
            self.has_bwd = False
        self.build_model(
            snapshot_shape=dataset.imshape,
            output_dims=dataset.imshape[-1],
            fwd_wt=args.forward,
            bwd_wt=args.backward,
            id_wt=args.identity,
            cons_wt=args.consistency,
            forward_steps=args.fwd_steps,
            backward_steps=args.bwd_steps,
            sizes=args.sizes,
            weight_decay=args.wd,
        )
    
    def build_model(self, snapshot_shape, output_dims, fwd_wt, bwd_wt, id_wt, 
            cons_wt, forward_steps, backward_steps, sizes, weight_decay):
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
        total_inpt_snapshots = backward_steps + 1 + forward_steps
        inpt = Input((total_inpt_snapshots,) + snapshot_shape)
        tf.print("inpt shape:", inpt.shape)
        current = inpt[:,backward_steps,:]
        tf.print("current shape:", current.shape)
        print("Autoencoder Input shape:", inpt.shape, "current shape:", current.shape)

        intermediate, bottleneck = sizes

        encoder = AutoencoderBlock((intermediate, intermediate, bottleneck), weight_decay,
            name="encoder")
        forward = Dense(bottleneck, use_bias=False, name="forward-dynamics",
            kernel_initializer=glorot_normal(), kernel_regularizer=regularizers.l2(weight_decay))
        decoder = AutoencoderBlock((intermediate, intermediate, output_dims), weight_decay,
            name="decoder")
        if self.has_bwd:
            # need to build fwd layer to access its weights in backward ConsistencyLayer
            inshape = encoder(current).shape
            backward = KoopmanConsistencyLayer(in_shape=inshape, 
                pair_layer=forward, cons_wt=cons_wt, 
                units=bottleneck, use_bias=False, weight_decay=weight_decay,
                name="backward-dynamics")

        # predict pred_steps into the future
        encoded_out = encoder(current)
        # outputs is list [earliest_backward_pred, ... latest_forward_pred]
        outputs = []
        # for each step size
        for s in range(1, forward_steps+1):
            # compute the forward and backward predictions
            x_f = encoded_out
            for _ in range(s):
                x_f = forward(x_f)
            outputs.append(decoder(x_f))
        
        for s in range(1, backward_steps+1):
            x_b = encoded_out
            for _ in range(s):
                x_b = backward(x_b)
            outputs.insert(0, decoder(x_b))

        model = Model(inputs=inpt, outputs=outputs)

        # Forward and Backward Dynamics Regularizers
        fwd_pred = tf.stack(outputs[backward_steps:], axis=1)
        fwd_true = inpt[:,backward_steps+1:,:]
        fwd_loss = fwd_wt * tf.reduce_mean((fwd_true - fwd_pred) ** 2)
        model.add_loss(fwd_loss)
        model.add_metric(fwd_loss, name="fwd_loss", aggregation="mean")

        if self.has_bwd:
            bwd_pred = tf.stack(outputs[:backward_steps], axis=1)
            bwd_true = inpt[:,:backward_steps,:]
            bwd_loss = bwd_wt * tf.reduce_mean((bwd_true - bwd_pred) ** 2)
            model.add_loss(bwd_loss)
            model.add_metric(bwd_loss, name="bwd_loss", aggregation="mean")

        # Encoder-Decoder Identity
        id_loss = id_wt * inverse_reg(current, encoder, decoder)
        model.add_loss(id_loss)
        model.add_metric(id_loss, name="id_loss", aggregation="mean")

        # model.summary()

        self.model = model
        self.encoder = encoder
        self.forward = forward
        self.decoder = decoder
        if self.has_bwd:
            self.backward = backward
        else:
            self.backward = None

    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=None)

    def make_run_name(self):
        args = self.args
        run_name = args.name + ".koop."
        run_name += "{}fs{}bs.".format(args.fwd_steps, args.bwd_steps)
        run_name += "f{}_b{}_i{}_c{}.".format(args.forward, args.backward, args.identity, args.consistency)
        run_name += self.run_name_common_suffix()
        return run_name

    def data_formatter(self, X, bwd_steps, fwd_steps):
        """
        slice X into sequences of 2*steps+1 snapshots for input to koopman autoencoder
        """
        out = []
        for i in range(bwd_steps, X.shape[0]-fwd_steps):
            out.append( X[i-bwd_steps:i+fwd_steps+1] )
        return np.array(out)

    def format_data(self):
        self.X = self.data_formatter(
            self.dataset.X, self.args.bwd_steps, self.args.fwd_steps)
        valX = self.data_formatter(
            self.dataset.Xtest, self.args.bwd_steps, self.args.fwd_steps)
        self.val_data = (valX, None)

    def train(self, callbacks):
        self.format_data()
        H = self.model.fit(
            x=self.X,
            y=None,
            # steps_per_epoch=3,
            batch_size=self.args.batchsize,
            epochs=self.args.epochs,
            callbacks=callbacks,
            validation_data=self.val_data,
            # validation_split=0.2,
            # validation_batch_size=args.batchsize,
            verbose=2,
        )
        return H
    
    def get_pipeline(self):
        return (self.encoder, self.forward, self.decoder)

    def save_eigenvals(self):
        run_name = self.make_run_name()
        output_eigvals(self.forward.weights[0], run_name, type_="forward")
        if self.has_bwd > 0:
            output_eigvals(self.backward.weights[0], run_name, type_="backward")


CUSTOM_OBJ_DICT.update({
    "KoopmanConsistencyLayer": KoopmanConsistencyLayer
})


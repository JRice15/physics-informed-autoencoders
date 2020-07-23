import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.initializers import glorot_normal, zeros, Initializer, RandomNormal
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
from src.dense_autoencoders import *
from src.conv_autoencoders import *
from src.output_results import *


class DynamicsInitializer(Initializer):

    def __init__(self, init_scale=0.99):
        self.scale = init_scale
    
    def __call__(self, shape, dtype=None):
        perm = [i for i in range(len(shape))]
        perm[0] = 1
        perm[1] = 0
        weight = RandomNormal(0, 1)(shape, dtype)
        U, S, V = np.linalg.svd(weight)
        V_T = np.transpose(V, axes=perm)
        return np.matmul(U, V_T) * self.scale
    
    def get_config(self):
        return {"init_scale": self.scale}


class InverseDynamicsInitializer(Initializer):
    """
    initialize a layers weights to be the (pseudo-)inverse of another layer's
    """

    def __init__(self, in_shape, pair_layer, conv_dynamics=False):
        self.in_shape = in_shape
        # build pair layer
        _ = pair_layer(Input(in_shape))
        self.pair_layer = pair_layer
        self.conv_dynamics = conv_dynamics
    
    def __call__(self, shape, dtype=None):
        pair_weights = self.pair_layer.weights[0]
        assert pair_weights.shape == shape
        perm = [i for i in range(len(shape))]
        perm[0] = 1
        perm[1] = 0
        inv_weights = tf.linalg.pinv(tf.transpose(pair_weights, perm=perm))
        return inv_weights

    def get_config(self):
        return {
            "pair_layer": self.pair_layer,
            "in_shape": self.in_shape,
            "conv_dynamics": self.conv_dynamics,
        }


class KoopmanConsistencyLayer(Layer):
    """
    enforce consistency of forward-backward predictions
    """

    def __init__(self, conv_dynamics, in_shape, pair_layer, cons_wt, units, name, weight_decay, 
            use_bias=False, **kwargs):
        super().__init__(name=name)
        self.conv_dynamics = conv_dynamics
        self.units = units
        self.in_shape = in_shape
        self.pair_layer = pair_layer
        self.cons_wt = cons_wt
        self.weight_decay = weight_decay

        kwargs["kernel_initializer"] = InverseDynamicsInitializer(in_shape, pair_layer, conv_dynamics=conv_dynamics)
        kwargs["kernel_regularizer"] = self.consistency_reg()
        kwargs["bias_initializer"] = zeros()

        if conv_dynamics:
            self.dyn_layer = Conv2D(1, 9, padding="same", name=name+"-conv", use_bias=False, **kwargs)
        else:
            self.dyn_layer = Dense(units=units, name=name+"-dense", use_bias=use_bias, **kwargs)
    

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
        
    def call(self, x):
        return self.dyn_layer(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "pair_layer": self.pair_layer,
            "cons_wt": self.cons_wt,
            "weight_decay": self.weight_decay,
            "in_shape": self.in_shape,
            "conv_dynamics": self.conv_dynamics,
        })
        return config


class KoopmanAutoencoder(BaseAE):

    def __init__(self, args, dataset=None):
        super().__init__(args, dataset)
        if args.bwd_steps > 0:
            self.has_bwd = True
        else:
            self.has_bwd = False
        self.build_enc_dec(args, dataset.input_shape)
        self.build_model(
            snapshot_shape=dataset.input_shape,
            output_dims=dataset.input_shape[-1],
            fwd_wt=args.forward,
            bwd_wt=args.backward,
            id_wt=args.identity,
            cons_wt=args.consistency,
            forward_steps=args.fwd_steps,
            backward_steps=args.bwd_steps,
            weight_decay=args.wd,
            conv_dynamics=args.conv_dynamics,
        )
    
    def build_enc_dec(self, args, input_shape):
        if args.convolutional:
            self.encoder = ConvEncoder(args.activation, args.depth, args.dilations, args.kernel_sizes, 
                args.filters, args.wd, conv_dynamics=args.conv_dynamics)

            # build encoder, get encoded (pre-flattened) shape
            _ = self.encoder(Input(input_shape))
            encoded_shape = self.encoder.encoded_shape

            # create decoder (with reversed kernels, reversed and inverse dilations)
            self.decoder = ConvDecoder(args.activation, args.depth, args.dilations[::-1], args.kernel_sizes[::-1],
                args.filters, args.wd, conv_dynamics=args.conv_dynamics, encoded_shape=encoded_shape, 
                target_shape=input_shape)

        else:
            intermediate, bottleneck = args.sizes

            self.encoder = DenseAutoencoderBlock((intermediate, intermediate, bottleneck), 
                args.wd, name="encoder")
            self.decoder = DenseAutoencoderBlock((intermediate, intermediate, input_shape[-1]), 
                args.wd, name="decoder")


    def build_model(self, snapshot_shape, output_dims, fwd_wt, bwd_wt, id_wt, 
            cons_wt, forward_steps, backward_steps, weight_decay, conv_dynamics):
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
        current = inpt[:,backward_steps,:]
        print("Autoencoder Input shape:", inpt.shape, "current shape:", current.shape)

        inshape = self.encoder(current).shape[1:] # remove batch size
        print("\n\ndynamics shape:", inshape)
        if conv_dynamics:
            self.forward = Conv2D(1, 5, padding="same", use_bias=False, name="forward-dynamics-conv",
                kernel_initializer=DynamicsInitializer(), kernel_regularizer=regularizers.l2(weight_decay))
        else:
            self.forward = Dense(inshape[-1], use_bias=False, name="forward-dynamics-dense",
                kernel_initializer=DynamicsInitializer(), kernel_regularizer=regularizers.l2(weight_decay))
        
        if self.has_bwd:
            self.backward = KoopmanConsistencyLayer(conv_dynamics=conv_dynamics, 
                in_shape=inshape, pair_layer=self.forward, cons_wt=cons_wt, 
                units=inshape[-1], use_bias=False, weight_decay=weight_decay,
                name="backward-dynamics")
        else:
            self.backward = None

        # predict pred_steps into the future
        encoded_out = self.encoder(current)
        # outputs is list [earliest_backward_pred, ... latest_forward_pred]
        outputs = []

        x_f = encoded_out
        for _ in range(forward_steps):
            x_f = self.forward(x_f)
            outputs.append(self.decoder(x_f))
        
        x_b = encoded_out
        for _ in range(backward_steps):
            x_b = self.backward(x_b)
            outputs.insert(0, self.decoder(x_b))

        self.model = Model(inputs=inpt, outputs=outputs)

        # Forward and Backward Dynamics Regularizers
        fwd_pred = tf.stack(outputs[backward_steps:], axis=1)
        fwd_true = inpt[:,backward_steps+1:,:]
        fwd_loss = fwd_wt * tf.reduce_mean((fwd_true - fwd_pred) ** 2)
        self.model.add_loss(fwd_loss)
        self.model.add_metric(fwd_loss, name="fwd_loss", aggregation="mean")

        if self.has_bwd:
            bwd_pred = tf.stack(outputs[:backward_steps], axis=1)
            bwd_true = inpt[:,:backward_steps,:]
            bwd_loss = bwd_wt * tf.reduce_mean((bwd_true - bwd_pred) ** 2)
            self.model.add_loss(bwd_loss)
            self.model.add_metric(bwd_loss, name="bwd_loss", aggregation="mean")

        # Encoder-Decoder Identity
        id_loss = id_wt * inverse_reg(current, self.encoder, self.decoder)
        self.model.add_loss(id_loss)
        self.model.add_metric(id_loss, name="id_loss", aggregation="mean")


    def compile_model(self, optimizer):
        self.model.compile(optimizer=optimizer, loss=None)

    def make_run_name(self):
        args = self.args
        run_name = args.name + ".koop."
        if args.convolutional:
            run_name += "conv."
        run_name += "{}fs{}bs.".format(args.fwd_steps, args.bwd_steps)
        run_name += "f{}_b{}_i{}_c{}.".format(args.forward, args.backward, args.identity, args.consistency)
        run_name += self.run_name_common_suffix()
        return run_name

    def data_formatter(self, X):
        """
        slice X into sequences of 2*steps+1 snapshots for input to koopman autoencoder
        """
        bwd = self.args.bwd_steps
        fwd = self.args.fwd_steps
        out = []
        for i in range(bwd, X.shape[0]-fwd):
            out.append( X[i-bwd:i+fwd+1] )
        return np.array(out)

    def format_data(self):
        self.X = self.data_formatter(self.dataset.X)
        valX = self.data_formatter(self.dataset.Xtest)
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
    "KoopmanConsistencyLayer": KoopmanConsistencyLayer,
    "DynamicsInitializer": DynamicsInitializer,
    "InverseDynamicsInitializer": InverseDynamicsInitializer,
})


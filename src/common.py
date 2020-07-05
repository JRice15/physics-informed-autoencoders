import argparse
import os
import json
import shutil
import re

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

def gather_args(model_type, num_sizes, defaults):
    """
    parse command line arguments and loads any presets
    Returns:
        args, run_name
    """
    parser = argparse.ArgumentParser(description="see Erichson et al's 'PHYSICS-INFORMED "
        "AUTOENCODERS FOR LYAPUNOV-STABLE FLUID FLOW PREDICTION' for context of "
        "greek-letter hyperparameters")

    parser.add_argument("--name",type=str,required=True,help="name of this training run")
    parser.add_argument("--lr",type=float,default=defaults.lr,help="learning rate")
    parser.add_argument("--epochs",type=int,default=defaults.epochs)
    parser.add_argument("--batchsize",type=int,default=defaults.batchsize)
    parser.add_argument("--sizes",type=int,default=defaults.sizes,nargs=num_sizes,help="encoder layer output widths in decreasing order of size")

    if model_type == "lyapunov":
        parser.add_argument("--lambd",type=float,default=defaults.lambda_,help="inverse regularizer weight")
        parser.add_argument("--kappa",type=float,default=defaults.kappa,help="stability regularizer weight")
        parser.add_argument("--gamma",type=float,default=defaults.gamma,help="stability regularizer steepness")
        parser.add_argument("--no-stability",action="store_true",default=False,help="use this flag for no stability regularization")
    elif model_type == "koopman":
        parser.add_argument("--pred-steps",default=defaults.pred_steps,help="number of forward and backward prediction steps to use in the loss")
        parser.add_argument("--identity",default=defaults.consistency,help="weight for decoder(encoder)==identity regularizer term")
        parser.add_argument("--forward",default=defaults.forward,help="weight for forward dynamics regularizer term")
        parser.add_argument("--backward",default=defaults.backward,help="weight for backward dynamics regularizer term")
        parser.add_argument("--consistency",default=defaults.consistency,help="weight for consistency regularizer term")

    parser.add_argument("--save",action="store_true",default=False,help="save these hyperparameters to a file, 'presets/<name>.<model>.json'")
    parser.add_argument("--load",action="store_true",default=False,help="load hyperparameters from a file in the 'presets/'")

    args = parser.parse_args()

    # echo args
    for k,v in args.__dict__.items():
        if v is not None:
            print("    " + k + ":", v)

    # allow hyperparamater saving/loading
    if args.save:
        print("Saving parameters...")
        os.makedirs("presets", exist_ok=True)
        path = "presets/" + args.name + "." + model_type + ".json"
        with open(path, "w") as f:
            json.dump(args.__dict__, f, indent=2)
    elif args.load:
        print("Loading parameters...")
        path = "presets/" + args.name + "." + model_type + ".json"
        with open(path, "r") as f:
            loaded = json.load(f)
        args.__dict__.update(loaded)

    return args


def make_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    os.makedirs("train_results", exist_ok=True)
    os.makedirs("test_results", exist_ok=True)
    os.makedirs("stats", exist_ok=True)
    if os.path.exists("logs"):
        print("Removing old Tensorboard logs...")
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)


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




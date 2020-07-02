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


class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.001
    epochs = 6000
    batchsize = 34
    lambda_ = 3 # inverse regularizer weight
    kappa = 3 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes = (40, 25, 15) # largest to smallest

def get_run_name(args):
    run_name = args.name + "."
    if args.no_stability:
        run_name += "nostability."
    else:
        run_name += "stability."
    run_name += "{}epochs_{}batchsize_{}lr.".format(args.epochs, args.batchsize, args.lr)
    run_name += "s{}_{}_{}.".format(args.s1, args.s2, args.s3)
    run_name += "l{}_k{}_g{}".format(args.lambd, args.kappa, args.gamma)
    return run_name



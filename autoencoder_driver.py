import os
import argparse

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam
from keras import metrics

from read_dataset import *
from shallow_autoencoder import *

class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.001
    epochs = 4000
    batchsize = 34
    lambda_ = 3 # inverse regularizer weight
    kappa = 0.1 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes=(40,25,15)

parser = argparse.ArgumentParser(description="see Erichson et al's 'PHYSICS-INFORMED "
    "AUTOENCODERS FOR LYAPUNOV-STABLE FLUID FLOW PREDICTION' for context of "
    "greek-letter hyperparameters")

parser.add_argument("--lr",type=float,default=Defaults.lr,help="learning rate")
parser.add_argument("--epochs",type=int,default=Defaults.epochs)
parser.add_argument("--batchsize",type=int,default=Defaults.batchsize)
parser.add_argument("--tag",type=str,default=None,help="tag to add to weights path to not overwrite the default path")
parser.add_argument("--lambd",type=float,default=Defaults.lambda_,help="(lambda) inverse regularizer weight")
parser.add_argument("--kappa",type=float,default=Defaults.kappa,help="stability regularizer weight")
parser.add_argument("--gamma",type=float,default=Defaults.gamma,help="stability regularizer steepness")
parser.add_argument("--no-stability",action="store_true",default=False,help="use this flag for no stability regularization")
parser.add_argument("--s1",type=int,default=Defaults.sizes[0],help="first encoder layer output width")
parser.add_argument("--s2",type=int,default=Defaults.sizes[1],help="second encoder layer output width")
parser.add_argument("--s3",type=int,default=Defaults.sizes[2],help="third encoder layer output width")

args = parser.parse_args()

print(args.__dict__)

os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)

# Read Data
X, Xtest = data_from_name("flow_cylinder")
datashape = X[0].shape

# Create Model
autoencoder = shallow_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    kappa=args.kappa,
    lambda_=args.lambd,
    gamma=args.gamma,
    no_stability=args.no_stability,
    sizes=(args.s1, args.s2, args.s3)
)
optimizer = Adam(args.lr)
autoencoder.compile(optimizer=optimizer, loss=mse, metrics=[metrics.MeanSquaredError()])

# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]

weights_path = "weights/weights"
if args.tag is not None:
    weights_path += "." + args.tag
if args.no_stability:
    weights_path += ".nostability"
weights_path += ".hdf5"

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=(args.lr / 16), verbose=1))
callbacks.append(ModelCheckpoint(weights_path, save_best_only=True, verbose=1))

print(X.shape, Y.shape)

history = autoencoder.fit(
    x=X, y=Y,
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=(Xtest, Ytest),
    verbose=2,
)

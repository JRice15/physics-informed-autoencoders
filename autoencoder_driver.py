import argparse
import json
import os
import re
import subprocess

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from read_dataset import *
from shallow_autoencoder import shallow_autoencoder
from output_results import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3


class Defaults:
    """
    namespace for defining arg defaults
    """
    lr = 0.001
    epochs = 4000
    batchsize = 34
    lambda_ = 3 # inverse regularizer weight
    kappa = 1 # stability regularizer weight
    gamma = 4 # stability regularizer steepness
    sizes = (40, 25, 15) # largest to smallest


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
parser.add_argument("--save",default=None,metavar="FILENAME",help="save these hyperparameters to a file, which will be placed in the 'presets/' directory")
parser.add_argument("--load",default=None,metavar="FILENAME",help="load hyperparameters from a file in the 'presets/'")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# echo args
for k,v in args.__dict__.items():
    if v is not None:
        print("    " + k + ":", v)

# allow hyperparamater saving/loading
if args.save is not None:
    os.makedirs("presets", exist_ok=True)
    path = "presets/" + re.sub(r"[^-_A-Za-z0-9]", "", args.save) + ".json"
    with open(path, "w") as f:
        json.dump(args.__dict__, f, indent=2)
elif args.load is not None:
    path = "presets/" + re.sub(r"[^-_A-Za-z0-9]", "", args.load) + ".json"
    with open(path, "r") as f:
        args.__dict__ = json.load(f)

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

optimizer = Adam(
    learning_rate=args.lr, 
    # clipvalue=5.0,
)
autoencoder.compile(optimizer=optimizer, loss=mse, metrics=[metrics.MeanSquaredError()])
    # experimental_run_tf_function=False)


# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]

run_name = ""
if args.tag is not None:
    run_name += args.tag + "."
if args.no_stability:
    run_name += "nostability."
run_name += "l{}_k{}_g{}.".format(args.lambd, args.kappa, args.gamma)

weights_path = "weights/weights." + run_name + "hdf5"

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 
        min_lr=(args.lr / 16), verbose=1),
    ModelCheckpoint(weights_path, save_best_only=True, verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=True, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    ImgWriter(autoencoder, run_name, Xtest, Ytest),
]

print("\n\n\nBegin Training")

history = autoencoder.fit(
    x=X, y=Y,
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=(Xtest, Ytest),
    verbose=2,
)


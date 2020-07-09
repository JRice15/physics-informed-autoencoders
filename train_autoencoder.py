import re
import time
import argparse
import os
import shutil
import json

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.losses import mse
from keras import metrics
from keras.models import Model
from keras.optimizers import Adam

from src.common import *
from src.lyapunov_autoencoder import *
from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

def make_dirs(dirname):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("train_results/" + dirname, exist_ok=True)
    os.makedirs("stats/" + dirname, exist_ok=True)
    if os.path.exists("logs"):
        print("Removing old Tensorboard logs...")
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)


# Get model type first
parser = argparse.ArgumentParser(description="Select model first")
parser.add_argument("--model",required=True,choices=["koopman","lyapunov"],help="name of the model to use")
args, unknown = parser.parse_known_args()

model_type = args.model
if model_type == "lyapunov":
    defaults = LyapunovAutoencoder.Defaults
    num_sizes = 3
elif model_type == "koopman":
    defaults = KoopmanAutoencoder.Defaults
    num_sizes = 2

# Parse Full Args
parser = argparse.ArgumentParser(description="see Erichson et al's 'PHYSICS-INFORMED "
    "AUTOENCODERS FOR LYAPUNOV-STABLE FLUID FLOW PREDICTION' for context of "
    "greek-letter hyperparameters")

parser.add_argument("--model",required=True,choices=["koopman","lyapunov"],help="name of the model to use")
parser.add_argument("--name",type=str,required=True,help="name of this training run")
parser.add_argument("--dataset",type=str,default="flow_cylinder",help="name of dataset")
parser.add_argument("--lr",type=float,default=defaults.lr,help="learning rate")
parser.add_argument("--wd",type=float,default=defaults.wd,help="weight decay weighting factor")
parser.add_argument("--gradclip",type=float,default=defaults.gradclip,help="gradient clipping by norm, or 0 for no gradclipping")
parser.add_argument("--seed",type=int,default=0,help="random seed")
parser.add_argument("--epochs",type=int,default=defaults.epochs)
parser.add_argument("--batchsize",type=int,default=defaults.batchsize)
parser.add_argument("--sizes",type=int,default=defaults.sizes,nargs=num_sizes,help="encoder layer output widths in decreasing order of size")

if model_type == "lyapunov":
    parser.add_argument("--lambd",type=float,default=defaults.lambda_,help="inverse regularizer weight")
    parser.add_argument("--kappa",type=float,default=defaults.kappa,help="stability regularizer weight")
    parser.add_argument("--gamma",type=float,default=defaults.gamma,help="stability regularizer steepness")
    parser.add_argument("--no-stability",action="store_true",default=False,help="use this flag for no stability regularization")
elif model_type == "koopman":
    parser.add_argument("--identity",type=float,default=defaults.consistency,help="weight for decoder(encoder)==identity regularizer term")
    parser.add_argument("--forward",type=float,default=defaults.forward,help="weight for forward dynamics regularizer term")
    parser.add_argument("--backward",type=float,default=defaults.backward,help="weight for backward dynamics regularizer term")
    parser.add_argument("--consistency",type=float,default=defaults.consistency,help="weight for consistency regularizer term")
    parser.add_argument("--fwd-steps",type=int,default=defaults.fwd_steps,help="number of forward prediction steps to use in the loss")
    parser.add_argument("--bwd-steps",type=int,default=defaults.bwd_steps,help="number of backward prediction steps to use in the loss")

parser.add_argument("--save",action="store_true",default=False,help="save these hyperparameters to a file, 'presets/<name>.<model>.json'")
parser.add_argument("--load",action="store_true",default=False,help="load hyperparameters from a file in the 'presets/'")

args = parser.parse_args()

# echo args
for k,v in args.__dict__.items():
    if v is not None:
        print("    " + k + ":", v, type(v))

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


set_seed(args.seed)

# Read Data
X, Xtest, data_formatter, imshape = data_from_name(args.dataset)
datashape = X[0].shape

if model_type == "lyapunov":
    autoencoder = LyapunovAutoencoder(args, datashape)
elif model_type == "koopman":
    autoencoder = KoopmanAutoencoder(args, datashape)

autoencoder.format_data(X, Xtest)

run_name = autoencoder.make_run_name()
make_dirs(run_name)

def lr_schedule(args):
    """
    reduce lr by 0.6 every (args.epochs // 5) epochs
    """
    def scheduler(epoch):
        exp = epoch // (args.epochs // 5)
        new_rate = args.lr * (0.4 ** exp)
        if epoch % (args.epochs // 5) == 0:
            print("LearningRateScheduler setting learning rate to {}".format(new_rate))
        return new_rate
    return scheduler

model_path = "models/model." + run_name + ".hdf5"

callbacks = [
    LearningRateScheduler(lr_schedule(args)),
    ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False, 
        verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    ImgWriter(pipeline=autoencoder.get_pipeline(), run_name=run_name, 
        Xtest=Xtest[:-1], Ytest=Xtest[1:], freq=args.epochs//5, data_formatter=data_formatter),
]

gradclip = None if args.gradclip == 0 else args.gradclip
optimizer = Adam(
    learning_rate=args.lr,
    clipnorm=gradclip
)

autoencoder.compile_model(optimizer)

print("\n\n\nBegin Training")

start_time = time.time()

H = autoencoder.train(callbacks)

print("Training took {0} minutes".format((time.time() - start_time)/60))
print("{0} seconds per epoch".format((time.time() - start_time)/args.epochs))

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 5

save_history(H, run_name, marker_step=marker_step)

autoencoder.save_eigenvals()

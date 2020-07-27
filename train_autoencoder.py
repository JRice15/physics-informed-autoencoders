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
                             ReduceLROnPlateau, TensorBoard, EarlyStopping)
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

# tf.compat.v1.disable_eager_execution()

def make_dirs(dirname):
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("train_results/" + dirname, exist_ok=True)
    os.makedirs("stats/" + dirname, exist_ok=True)
    if os.path.exists("logs"):
        print("Removing old Tensorboard logs...")
        shutil.rmtree("logs")
    os.makedirs("logs", exist_ok=True)


# Get model and dataset first
parser = argparse.ArgumentParser(description="Select model first")
parser.add_argument("--model",required=True,choices=["koopman","lyapunov"],help="name of the model to use")
parser.add_argument("--dataset",required=True,type=str,help="name of dataset")
parser.add_argument("--convolutional",action="store_true",default=False,help="create a convolutional model")
parser.add_argument("--no-basemap",action="store_true",default=False,help="do not use basemap (cannot write sst images)")

args, unknown = parser.parse_known_args()

model_type = args.model

# Read Data
dataset = data_from_name(args.dataset, (not args.convolutional), no_basemap=args.no_basemap)

# Load defaults
if args.convolutional:
    defaults_file = "presets/conv-default.{}.{}.json".format(model_type, dataset.dataname)
else:
    defaults_file = "presets/orig-paper.{}.{}.json".format(model_type, dataset.dataname)
try:
    with open(defaults_file, "r") as f:
        defaults = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Defaults file for this model and dataset configuration could not be found. Create "
        "a file '{}' with the desired presets".format(defaults_file))

if args.convolutional:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth",type=int,default=defaults["depth"],help="depth of convolutional network")
    depth_args, unknown = parser.parse_known_args()
    num_sizes = depth_args.depth
else:
    if model_type == "lyapunov":
        num_sizes = 3
    elif model_type == "koopman":
        num_sizes = 2

# Parse Full Args
parser = argparse.ArgumentParser(description="see Erichson et al's 'PHYSICS-INFORMED "
    "AUTOENCODERS FOR LYAPUNOV-STABLE FLUID FLOW PREDICTION' for context of "
    "greek-letter hyperparameters")

parser.add_argument("--model",required=True,choices=["koopman","lyapunov"],help="name of the model to use")
parser.add_argument("--name",type=str,required=True,help="name of this training run")
parser.add_argument("--convolutional",action="store_true",default=False,help="create a convolutional version of the model")
parser.add_argument("--conv-dynamics",action="store_true",default=False,help="use a convolutional dynamics layer")
parser.add_argument("--dataset",type=str,default="flow_cylinder",help="name of dataset")
parser.add_argument("--lr",type=float,default=defaults["lr"],help="learning rate")
parser.add_argument("--wd",type=float,default=defaults["wd"],help="weight decay weighting factor")
parser.add_argument("--gradclip",type=float,default=defaults["gradclip"],help="gradient clipping by norm, or 0 for no gradclipping")
parser.add_argument("--seed",type=int,default=0,help="random seed")
parser.add_argument("--epochs",type=int,default=defaults["epochs"])
parser.add_argument("--batchsize",type=int,default=defaults["batchsize"])
parser.add_argument("--activation",type=str,default="tanh",help="name of activation to use")

parser.add_argument("--tboard",action="store_true",default=False,help="run tensorboard")
parser.add_argument("--summary",action="store_true",default=False,help="show model summary")
parser.add_argument("--no-earlystopping",action="store_true",default=False,help="do not use early stopping")
parser.add_argument("--no-basemap",action="store_true",default=False,help="do not use basemap (cannot write sst images)")

if args.convolutional:
    parser.add_argument("--depth",type=int,default=defaults["depth"],help="depth of convolutional network")
    parser.add_argument("--dilations",type=int,default=defaults["dilations"],nargs=num_sizes,help="encoder layer conv dilations in order")
    parser.add_argument("--kernel-sizes",type=int,default=defaults["kernel_sizes"],nargs=num_sizes,help="encoder layer kernel sizes in order")
    parser.add_argument("--filters",type=int,default=defaults["filters"],nargs=num_sizes-1,help="encoder layer filters in order, except last (which is always 1)")
else:
    parser.add_argument("--sizes",type=int,default=defaults["sizes"],nargs=num_sizes,help="encoder layer output widths in decreasing order of size")

if model_type == "lyapunov":
    parser.add_argument("--lambd",type=float,default=defaults["lambd"],help="inverse regularizer weight")
    parser.add_argument("--kappa",type=float,default=defaults["kappa"],help="stability regularizer weight")
    parser.add_argument("--gamma",type=float,default=defaults["gamma"],help="stability regularizer steepness")
    parser.add_argument("--no-stability",action="store_true",default=False,help="use this flag for no stability regularization")
elif model_type == "koopman":
    parser.add_argument("--identity",type=float,default=defaults["identity"],help="weight for decoder(encoder)==identity regularizer term")
    parser.add_argument("--forward",type=float,default=defaults["forward"],help="weight for forward dynamics regularizer term")
    parser.add_argument("--backward",type=float,default=defaults["backward"],help="weight for backward dynamics regularizer term")
    parser.add_argument("--consistency",type=float,default=defaults["consistency"],help="weight for consistency regularizer term")
    parser.add_argument("--fwd-steps",type=int,default=defaults["fwd_steps"],help="number of forward prediction steps to use in the loss")
    parser.add_argument("--bwd-steps",type=int,default=defaults["bwd_steps"],help="number of backward prediction steps to use in the loss")

parser.add_argument("--save",action="store_true",default=False,help="save these hyperparameters to a file, 'presets/<name>.<model>.json'")
parser.add_argument("--load",action="store_true",default=False,help="load hyperparameters from a file in the 'presets/'")

args = parser.parse_args()

if args.conv_dynamics and not args.convolutional:
    raise ValueError("Convolutional dynamics require a convolutional encoder/decoder (add the --convolutional flag)")

# echo args
for k,v in args.__dict__.items():
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

if model_type == "lyapunov":
    autoencoder = LyapunovAutoencoder(args, dataset)
elif model_type == "koopman":
    autoencoder = KoopmanAutoencoder(args, dataset)

run_name = autoencoder.make_run_name()
make_dirs(run_name)

def lr_schedule(args):
    """
    reduce lr by 0.6 every (args.epochs // 5) epochs
    """
    def scheduler(epoch):
        if epoch < 50:
            return args.lr            
        exp = epoch // (args.epochs // 5) + 1
        new_rate = args.lr * (0.4 ** exp)
        if epoch % (args.epochs // 5) == 0 or (epoch == 50):
            print("LearningRateScheduler setting learning rate to {}".format(new_rate))
        return new_rate
    return scheduler

model_path = "models/model." + run_name + ".hdf5"

callbacks = [
    History(),
    LearningRateScheduler(lr_schedule(args)),
    ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False, 
        verbose=1, period=min(20, args.epochs//5)),
    ImgWriter(pipeline=autoencoder.get_pipeline(), run_name=run_name, 
        dataset=dataset, freq=args.epochs//5),
]
if not args.no_earlystopping:
    callbacks.append(
        EarlyStopping(min_delta=1e-5, patience=round(args.epochs // 5 * 1.05), mode="min",
            verbose=1))
if args.tboard:
    callbacks.append(TensorBoard(histogram_freq=100, write_graph=True, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100))

gradclip = None if args.gradclip == 0 else args.gradclip
optimizer = Adam(
    learning_rate=args.lr,
    clipnorm=gradclip
)

autoencoder.compile_model(optimizer)

if args.summary:
    import pydot
    # summarize
    autoencoder.model.summary()
    # plot
    keras.utils.pydot = pydot
    layers = autoencoder.model._layers
    autoencoder.model._layers = [i for i in layers if isinstance(i, Layer)]
    keras.utils.plot_model(
        autoencoder.model, show_shapes=True
    )
    autoencoder.model._layers = layers

# output AE structure
vis_model(autoencoder.model)

print("\n\n\nBegin Training")

start_time = time.time()

try:
    H = autoencoder.train(callbacks)
except KeyboardInterrupt:
    for c in callbacks: c.on_train_end()
    H = callbacks[0]

secs = time.time() - start_time
epochs_ran = len(H.history["loss"])
mins = secs / 60
secs_per_epoch = secs / epochs_ran
print("Training took {0} minutes".format(mins))
print("{0} seconds per epoch".format(secs_per_epoch))

marker_step = epochs_ran // 5
save_history(H, run_name, secs, marker_step=marker_step)

autoencoder.save_eigenvals()

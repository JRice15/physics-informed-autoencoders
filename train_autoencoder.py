import argparse
import json
import os
import re
import shutil

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from src.read_dataset import *
from src.lyapunov_autoencoder import lyapunov_autoencoder
from src.output_results import *
from src.common import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3


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
os.makedirs("train_results", exist_ok=True)
if os.path.exists("logs"):
    print("Removing old Tensorboard logs...")
    shutil.rmtree("logs")
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
        loaded = json.load(f)
    args.__dict__.update(loaded)    


# Read Data
X, Xtest = data_from_name("flow_cylinder")
datashape = X[0].shape

# Create Model
autoencoder = lyapunov_autoencoder(
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


run_name = get_run_name(args)
weights_path = "weights/weights." + run_name + ".hdf5"


# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]


def lr_schedule(epoch):
    max_divisor = 5
    divisor = epoch // 1000
    new_rate = args.lr / (2 ** min(divisor, max_divisor))
    if epoch % 1000 == 0:
        print("LearningRateScheduler setting learning rate to {}".format(new_rate))
    return new_rate

callbacks = [
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 
    #     min_lr=(args.lr / 16), verbose=1),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint(weights_path, save_best_only=True, verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    ImgWriter(autoencoder, run_name, Xtest, Ytest),
]

print("\n\n\nBegin Training")

H = autoencoder.fit(
    x=X, y=Y,
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=(Xtest, Ytest),
    verbose=2,
)

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

save_history(H, run_name, marker_step=marker_step)


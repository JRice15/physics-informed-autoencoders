import argparse
import json
import os
import re
import shutil
import re

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model
from keras import Input
import tkinter as tk
from tkinter import filedialog

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
parser.add_argument("--load",default=None,metavar="FILENAME",help="load hyperparameters from a file in the 'presets/'")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)
os.makedirs("test_results", exist_ok=True)


# echo args
for k,v in args.__dict__.items():
    if v is not None:
        print("    " + k + ":", v)

# allow hyperparamater loading
if args.load is not None:
    path = "presets/" + re.sub(r"[^-_A-Za-z0-9]", "", args.load) + ".json"
    with open(path, "r") as f:
        args.__dict__ = json.load(f)

# Read Data
X, Xtest = data_from_name("flow_cylinder")
datashape = X[0].shape
data = np.concatenate([X, Xtest], axis=0)

# Create Model
models = lyapunov_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    kappa=args.kappa,
    lambda_=args.lambd,
    gamma=args.gamma,
    no_stability=args.no_stability,
    sizes=(args.s1, args.s2, args.s3)
)
autoencoder, encoder, dynamics, decoder = models

def run_test(weights_path, data, name, num_steps=50):

    autoencoder.load_weights(weights_path)

    num_snapshots = data.shape[0]
    data = np.reshape(data, (1, num_snapshots, -1))
    tfdata = tf.convert_to_tensor(data)

    error = []

    print("\n")
    print(weights_path)
    for step in range(1, num_steps+1):
        step_mse = []
        for i in range(num_snapshots - num_steps):
            snapshot = tfdata[:,i,:]

            x = encoder(snapshot)
            for _ in range(step):
                x = dynamics(x)
            pred = decoder(x).numpy()

            true = data[:,i+step,:]
            mse = np.mean((true - pred) ** 2)
            step_mse.append(mse)

            if step % 10 == 0 and i == 7:
                write_im(pred, title=str(step) + " steps prediction", 
                    filename=name + "_step" + str(step), directory="test_results" )
                write_im(true, title=str(step) + " steps ground truth", 
                    filename="truth_step" + str(step), directory="test_results")
        
        mean_mse = np.mean(step_mse)
        print(step, "steps MSE:", mean_mse)
        error.append(mean_mse)
    
    print("")
    return error


run_name = get_run_name(args)
weights_path = "weights/weights." + run_name + ".hdf5"

name1 = re.sub(r"\s", "_", input("Name for these initial weights: "))

# Select weights 2

yn = input("Compare to another run? [y/n]: ")
if yn.lower().strip() == "y":
    root = tk.Tk()
    root.withdraw()

    print("Select weights file to compare to...")
    weights2_path = filedialog.askopenfilename(initialdir="./weights/", title="select .hdf5 weights file")

    name2 = re.sub(r"\s", "_", input("Name for second chosen weights: "))

    error1 = run_test(weights_path, data, name1)
    error2 = run_test(weights2_path, data, name2)
    dnames = (name1, name2)
else:
    error1 = run_test(weights_path, data, name1)
    error2 = None
    dnames = (name1,None)

xrange = list(range(len(error1)))
make_plot(xrange=xrange, data=(error1, error2), dnames=dnames, title="MSE for Multi-Step Predictions", 
    mark=0, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(len(error1) // 6))

if dnames[1] is None:
    plt.savefig("test_results/multistep_mse_" + dnames[0] + ".png")
else:
    plt.savefig("test_results/multistep_mse_" + dnames[0] + "_vs_" + dnames[1] + ".png")
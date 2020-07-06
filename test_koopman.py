import argparse
import json
import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model

from src.common import *
from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

make_dirs()
args = gather_args("koopman", 2, Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest, imshape = data_from_name("flow_cylinder")
datashape = X[0].shape
data = np.concatenate([X, Xtest], axis=0)

# Create Model
models = koopman_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    fwd_wt=args.forward,
    bwd_wt=args.backward,
    id_wt=args.identity,
    cons_wt=args.consistency,
    forward_steps=args.fwd_steps,
    backward_steps=args.bwd_steps,
    sizes=args.sizes,
)
if args.bwd_steps > 0:
    autoencoder, encoder, dynamics, backward_dyn, decoder = models
else:
    autoencoder, encoder, dynamics, decoder = models

def run_test(weights_path, data, name, num_steps=50):
    """
    test a set of weights with multi-step prediction
    """
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
                write_im(pred, imshape, title=str(step) + " steps prediction", 
                    filename=name + "_step" + str(step), directory="test_results" )
                write_im(true, imshape, title=str(step) + " steps ground truth", 
                    filename="truth_step" + str(step), directory="test_results")
        
        mean_mse = np.mean(step_mse)
        print(step, "steps MSE:", mean_mse)
        error.append(mean_mse)
    
    print("")
    return error


run_name = make_run_name(args)
weights_path = "weights/weights." + run_name + "hdf5"

# Select weights 2

yn = input("Compare to another run? [y/n]: ")
if yn.lower().strip() == "y":
    root = tk.Tk()
    root.withdraw()

    print("Select weights file to compare to...")
    weights2_path = filedialog.askopenfilename(initialdir="./weights/", title="select .hdf5 weights file")

    name2 = re.sub(r"\s", "_", input("Name for second chosen weights: "))

    error1 = run_test(weights_path, data, args.name)
    error2 = run_test(weights2_path, data, name2)
    dnames = (args.name, name2)
else:
    error1 = run_test(weights_path, data, args.name)
    error2 = None
    dnames = (args.name,None)

xrange = list(range(len(error1)))
make_plot(xrange=xrange, data=(error1, error2), dnames=dnames, title="MSE for Multi-Step Predictions", 
    mark=0, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(len(error1) // 6))

if dnames[1] is None:
    plt.savefig("test_results/multistep_mse_" + dnames[0] + ".png")
else:
    plt.savefig("test_results/multistep_mse_" + dnames[0] + "_vs_" + dnames[1] + ".png")

print("Results have been save to 'test_results/'")

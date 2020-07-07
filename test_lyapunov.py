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
from src.lyapunov_autoencoder import *
from src.output_results import *
from src.read_dataset import *
from src.testing import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

args = gather_args("lyapunov", 3, Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest, imshape = data_from_name("flow_cylinder")
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
    sizes=args.sizes,
    weight_decay=args.wd,
)


run_name = make_run_name(args)
weights_path = "weights/weights." + run_name + ".hdf5"

# Select weights 2

yn = input("Compare to another run? [y/n]: ")
if yn.lower().strip() == "y":
    root = tk.Tk()
    root.withdraw()

    print("Select weights file to compare to...")
    weights2_path = filedialog.askopenfilename(initialdir="./weights/", title="select .hdf5 weights file")

    name2 = re.sub(r"\s", "_", input("Name for second chosen weights: "))

    error1 = run_test(models, weights_path, data, args.name)
    error2 = run_test(models, weights2_path, data, name2)
    dnames = (args.name, name2)
else:
    error1 = run_test(models, weights_path, data, args.name)
    error2 = None
    dnames = (args.name,None)

xrange = list(range(len(error1)))
make_plot(xrange=xrange, data=(error1, error2), dnames=dnames, title="MSE for Multi-Step Predictions", 
    mark=0, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(len(error1) // 6))

if dnames[1] is None:
    plt.savefig("test_results/" + dnames[0] + "_multistep_mse.png")
else:
    plt.savefig("test_results/" + dnames[0] + "_vs_" + dnames[1] + "_multistep_mse.png")

print("Results have been save to 'test_results/'")

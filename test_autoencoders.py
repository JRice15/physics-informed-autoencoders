import argparse
import json
import os
import re
import shutil
import tkinter as tk
import time
from tkinter import filedialog

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model

from src.common import *
from src.autoencoders import *
from src.lyapunov_autoencoder import *
from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

parser = argparse.ArgumentParser()

parser.add_argument("--pred-steps",type=int,default=50,help="number of timesteps to predict")
parser.add_argument("--file",default=None,help="file with weights paths to compare. Each line should be: '<name><tab-character><weights path>'")
parser.add_argument("--seed",type=int,default=0)

args = parser.parse_args()

set_seed(args.seed)

def get_pipeline(model):
    """
    returns [encoder, dynamics, decoder] layers
    """
    encoders = [i for i in model.layers if "encoder" in i.name]
    if len(encoders) != 1:
        raise ValueError("{} encoder layers found: {}".format(len(encoders), encoders))
    dynamics = [i for i in model.layers if "dynamics" in i.name]
    if len(dynamics) != 1:
        dynamics_2 = [i for i in dynamics if "forward" in i.name]
        if len(dynamics_2) != 1:
            raise ValueError("{} dynamics layers found, could not resolve forward/backward: {}".format(len(dynamics), dynamics))
        dynamics = dynamics_2
    decoders = [i for i in model.layers if "decoder" in i.name]
    if len(decoders) != 1:
        raise ValueError("{} decoder layers found: {}".format(len(decoders), decoders))
    return encoders + dynamics + decoders


# Read Data
X, Xtest, imshape = data_from_name("flow_cylinder")
datashape = X[0].shape
data = np.concatenate([X, Xtest], axis=0)


def run_name_from_model_path(model_path):
    model_path = re.sub(r".*/models/model\.", "", model_path)
    model_path = re.sub(r"\.hdf5", "", model_path)
    return model_path


def run_one_test(model_path, data, num_steps):
    """
    test a set of weights with multi-step prediction
    """
    autoencoder = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJ_DICT)
    encoder, dynamics, decoder = get_pipeline(autoencoder)

    dirname = run_name_from_model_path(model_path).strip(".")
    os.makedirs("test_results/" + dirname, exist_ok=True)

    num_snapshots = data.shape[0]
    data = np.reshape(data, (1, num_snapshots, -1))
    tfdata = tf.convert_to_tensor(data)

    error = []

    print("\n")
    print(model_path)
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

            if (step % 10 == 0 or step in (1,3,5)) and i == 7:
                write_im(pred, title=str(step) + " steps prediction", 
                    filename="pred_step" + str(step), directory="test_results/"+dirname )
                if not os.path.exists("test_results/truth/truth_step" + str(step) + ".png"):
                    os.makedirs("test_results/truth",exist_ok=True)
                    write_im(true, title=str(step) + " steps ground truth", 
                        filename="truth_step" + str(step), directory="test_results/truth")
        
        mean_mse = np.mean(step_mse)
        print(step, "steps MSE:", mean_mse)
        error.append(mean_mse)
    
    print("")
    return error


if args.file is not None:
    # File mode
    with open(args.file, "r") as f:
        lines = f.readlines()
    lines = [i.split("\t",maxsplit=1) for i in lines]
    names = [i[0].strip() for i in lines]
    paths = [i[1].strip() for i in lines]
    
else:
    # User interface mode
    paths = []
    names = []

    root = tk.Tk()
    root.withdraw()

    print("Select model files to compare, and cancel when complete...")
    time.sleep(0.5)
    while True:
        model_path = filedialog.askopenfilename(initialdir="models/", title="select a .hdf5 model file, or cancel to be done")
        if model_path == "":
            break
        print(model_path)
        name = re.sub(r"\s", "_", input("Name for this model: "))
        paths.append(model_path)
        names.append(name)


results = []

for p in paths:
    error = run_one_test(p, data, args.pred_steps)
    results.append(error)

xrange = list(range(len(results[0])))
make_plot(xrange=xrange, data=tuple(results), dnames=names, title="MSE of Multi-Step Predictions", 
    mark=0, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(len(results[0]) // 6))

fullname = "_vs_".join(names)
plt.savefig("test_results/" + fullname + ".multistep_mse.png")

print("Results have been save to 'test_results/'")


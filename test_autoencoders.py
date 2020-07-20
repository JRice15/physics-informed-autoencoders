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
from src.dense_autoencoders import *
from src.conv_autoencoders import *
from src.lyapunov_autoencoder import *
from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

parser = argparse.ArgumentParser()

parser.add_argument("--name",required=True,help="name of this test")
parser.add_argument("--dataset",required=True,help="name of the dataset to use")
parser.add_argument("--pred-steps",type=int,default=50,help="number of timesteps to predict")
parser.add_argument("--file",default=None,help="file with weights paths to compare. Each line should be: '<name><tab-character><weights path>'")
parser.add_argument("--convolutional",action="store_true",default=False,help="whether to test convolutional models")
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--quick",action="store_true",default=False,help="whether to just test steps 1,3,5,10,20,30,...")
parser.add_argument("--load-last",action="store_true",default=False,help="reload data of last training run")

args = parser.parse_args()

if args.load_last and args.file:
    raise ValueError("load-last and file args are incompatible")

set_seed(args.seed)
os.makedirs("test_results/truth",exist_ok=True)

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
    pipeline = encoders + dynamics + decoders
    for i in pipeline:
        print(i.name)
    return pipeline


# Read Data
dataset = data_from_name(args.dataset, (not args.convolutional))
data = np.concatenate([dataset.X, dataset.Xtest], axis=0)

print("data shape:", data.shape)

def run_name_from_model_path(model_path):
    model_path = re.sub(r".*/models/model\.", "", model_path)
    model_path = re.sub(r"\.hdf5", "", model_path)
    return model_path


def run_one_test(model_path, data, num_steps, step_arr):
    """
    test a set of weights with multi-step prediction
    """
    autoencoder = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJ_DICT)
    encoder, dynamics, decoder = get_pipeline(autoencoder)

    dirname = run_name_from_model_path(model_path)
    os.makedirs("test_results/" + dirname, exist_ok=True)

    shape = data.shape
    num_snapshots = shape[0]
    data = data.reshape((1,) + shape)
    print("data shape:", data.shape)
    tfdata = tf.convert_to_tensor(data)

    mse_min = []
    mse_max = []
    mse_avg = []
    relpred_min = []
    relpred_max = []
    relpred_avg = []

    print("\n")
    print(model_path)

    x = []
    for i in range(num_snapshots - num_steps):
        snapshot = tfdata[:,i,...]
        if i == 0: print("snapshot shape:", snapshot.shape)
        x.append(encoder(snapshot))

    prev_step = 0
    for step in step_arr:
        step_mse = []
        step_relpred_err = []
        for i in range(num_snapshots - num_steps):

            # next step(s)
            for _ in range(step - prev_step):
                x[i] = dynamics(x[i])
            pred = decoder(x[i]).numpy()
            true = data[:,i+step,:]

            mse = np.mean((true - pred) ** 2)
            step_mse.append(mse)

            relpred_err = np.linalg.norm(pred - true) / np.linalg.norm(true)
            step_relpred_err.append(relpred_err)

            if (step % 10 == 0 or step in (1,3,5)) and i == dataset.write_index:
                dataset.write_im(pred, title=str(step) + " steps prediction", 
                    filename="pred_step" + str(step), directory="test_results/"+dirname )
                truthfile = "test_results/truth/" + dataset.dataname + ".truth_step" + str(step) + ".png"
                if not os.path.exists(truthfile):
                    dataset.write_im(true, title=str(step) + " steps ground truth", 
                        filename=dataset.dataname + ".truth_step" + str(step), directory="test_results/truth")
        
        prev_step = step

        mean_mse = np.mean(step_mse)
        mean_relpred_err = np.mean(step_relpred_err)
        print(step, "steps MSE:", mean_mse, "relative error:", mean_relpred_err)
        mse_avg.append(mean_mse)
        relpred_avg.append(mean_relpred_err)

        mse_min.append(np.percentile(step_mse, 5))
        mse_max.append(np.percentile(step_mse, 95))
        relpred_min.append(np.percentile(step_relpred_err, 5))
        relpred_max.append(np.percentile(step_relpred_err, 95))
        
    return mse_min, mse_max, mse_avg, relpred_min, relpred_max, relpred_avg


if args.file is not None:
    # File mode
    with open(args.file, "r") as f:
        lines = f.readlines()
    lines = [i.split("\t",maxsplit=1) for i in lines]
    names = [i[0].strip() for i in lines]
    paths = [i[1].strip() for i in lines]
    
elif not args.load_last:
    # User interface mode
    paths = []
    names = []

    root = tk.Tk()
    root.withdraw()

    print("Select model files to compare, and cancel when complete")
    print("Make sure the model you select is for the dataset '{}', otherwise there will likely be an error".format(args.dataset))
    time.sleep(0.5)
    while True:
        model_path = filedialog.askopenfilename(initialdir="models/", title="select a .hdf5 model file, or cancel to be done")
        if model_path == "":
            break
        print(model_path)
        name = re.sub(r"\s", "_", input("Name for this model: "))
        paths.append(model_path)
        names.append(name)

if args.quick:
    step_arr = [1,3,5] + list(range(10,args.pred_steps+1,10))
else:
    step_arr = range(1, args.pred_steps+1)


if args.load_last:
    mse_avgs, mse_errbounds, relpred_avgs, relpred_errbounds, names = np.load("test_results/lastdata.npy", allow_pickle=True)
else:
    mse_avgs = []
    mse_errbounds = []
    relpred_avgs = []
    relpred_errbounds = []

    for p in paths:
        m_min, m_max, m_avg, r_min, r_max, r_avg = run_one_test(p, data, args.pred_steps, step_arr)
        mse_avgs.append(m_avg)
        mse_errbounds.append( (m_min, m_max) )
        relpred_avgs.append(r_avg)
        relpred_errbounds.append( (r_min, r_max) )
    
    np.save("test_results/lastdata.npy", [mse_avgs, mse_errbounds, relpred_avgs, relpred_errbounds, names])


fullname = args.name + "." + dataset.dataname
if args.convolutional:
    fullname += ".conv"
fullname += "." + str(args.pred_steps)
if args.quick:
    fullname += ".q"


final_mses = [i[-1] for i in relpred_avgs]
mark = final_mses.index(min(final_mses))

# MSE
make_plot(xrange=step_arr, data=tuple(mse_avgs), dnames=names, title="Prediction MSE -- " + args.dataset, 
    mark=mark, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=mse_errbounds,
    fillbetween_desc="w/ 90% confidence", ylim=1)
plt.savefig("test_results/" + fullname + ".multistep_mse.w_confidence.png")
plt.clf()

make_plot(xrange=step_arr, data=tuple(mse_avgs), dnames=names, title="Prediction MSE -- " + args.dataset, 
    mark=mark, axlabels=("steps", "mean squared error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=None,
    fillbetween_desc="", ylim=1)
plt.savefig("test_results/" + fullname + ".multistep_mse.png")
plt.clf()

# Relative Error
make_plot(xrange=step_arr, data=tuple(relpred_avgs), dnames=names, title="Prediction Relative Error -- " + args.dataset, 
    mark=mark, axlabels=("steps", "relative error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=relpred_errbounds,
    fillbetween_desc="w/ 90% confidence", ylim=1)
plt.savefig("test_results/" + fullname + ".multistep_relpred_err.w_confidence.png")
plt.clf()

make_plot(xrange=step_arr, data=tuple(relpred_avgs), dnames=names, title="Prediction Relative Error -- " + args.dataset, 
    mark=mark, axlabels=("steps", "relative error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=None,
    fillbetween_desc="", ylim=1)
plt.savefig("test_results/" + fullname + ".multistep_relpred_err.png")
plt.clf()

print("Results have been save to 'test_results/'")


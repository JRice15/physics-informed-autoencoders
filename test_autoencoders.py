import argparse
import json
import os
import re
import shutil
import tkinter as tk
import time
from tkinter import filedialog
import copy

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model

from src.common import *
from src.baselines import *
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
parser.add_argument("--pred-steps",type=int,default=180,help="number of timesteps to predict")
parser.add_argument("--file",default=None,help="file with weights paths to compare. Each line should be: '<name><tab-character><weights path>'")
parser.add_argument("--convolutional",action="store_true",default=False,help="whether to test convolutional models")
parser.add_argument("--seed",type=int,default=0)
parser.add_argument("--no-quick",action="store_true",default=False,help="whether to just test steps 1,3,5,10,20,30,...")
parser.add_argument("--load-last",action="store_true",default=False,help="reload data of last training run")
parser.add_argument("--overwrite",action="store_true",default=False)
parser.add_argument("--mask",action="store_true",default=False,help="whether to mask out land")

args = parser.parse_args()

if args.load_last and args.file:
    raise ValueError("load-last and file args are incompatible")

# echo args
for k,v in args.__dict__.items():
    print("    " + k + ":", v, type(v))

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
dataset = data_from_name(args.dataset, flat=(not args.convolutional), full_test=True)
data = dataset.Xtest

print("data shape:", data.shape)

def run_name_from_model_path(model_path):
    model_path = re.sub(r".*/models/model\.", "", model_path)
    model_path = re.sub(r"\.hdf5", "", model_path)
    return model_path



if args.mask:
    try:
        mask = dataset.mask
    except AttributeError:
        raise ValueError("This dataset does not support masking. Use an SST dataset")
else:
    mask = None

try:
    de_scale = dataset.de_scale
    descale_units = dataset.de_scale_units
    has_descale = True
except AttributeError:
    has_descale = False
    descale_units = "n/a"

def run_one_test(model_path, data, tfdata, num_steps, step_arr):
    """
    test a set of weights with multi-step prediction
    """
    autoencoder = keras.models.load_model(model_path, custom_objects=CUSTOM_OBJ_DICT)

    vis_model(autoencoder)
    encoder, dynamics, decoder = get_pipeline(autoencoder)

    dirname = run_name_from_model_path(model_path)
    os.makedirs("test_results/preds/" + dirname, exist_ok=True)

    shape = data.shape
    num_snapshots = shape[1]

    mse_min = []
    mse_max = []
    mse_avg = []
    mae_min = []
    mae_max = []
    mae_avg = []
    relpred_min = []
    relpred_max = []
    relpred_avg = []
    dmae_min = []
    dmae_avg = []
    dmae_max = []

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
        step_mae = []
        step_relpred_err = []
        step_dmae = []
        for i in range(num_snapshots - num_steps):

            # next step(s)
            for _ in range(step - prev_step):
                x[i] = dynamics(x[i])
            full_pred = decoder(x[i]).numpy().squeeze()
            full_true = data[:,i+step,:].squeeze()
            if has_descale:
                d_pred = de_scale(full_pred)
                d_true = de_scale(full_true)
                if mask is not None:
                    d_pred = d_pred[mask.flatten()]
                    d_true = d_true[mask.flatten()]
                d_diff = d_pred - d_true
                d_mae = np.mean(np.abs(d_diff))
                step_dmae.append(d_mae)
            if mask is not None:
                pred = full_pred[mask]
                true = full_true[mask]
            else:
                pred = full_pred
                true = full_true

            diff = pred - true
            mse = np.mean(diff ** 2)
            step_mse.append(mse)

            mae = np.mean(np.abs(diff))
            step_mae.append(mae)

            relpred_err = np.linalg.norm(diff) / np.linalg.norm(true)
            step_relpred_err.append(relpred_err)

            if (step % 10 == 0 or step in (1,3,5)) and (i == dataset.write_index or i in dataset.test_write_inds):
                dataset.write_im(full_pred, title=str(step) + " steps prediction", 
                    filename="index" + str(i) + ".pred_step" + str(step), directory="test_results/preds/"+dirname)
                truthfile = "test_results/truth/" + dataset.dataname + ".index" + str(i) + ".truth_step" + str(step) + ".png"
                if args.overwrite or not os.path.exists(truthfile):
                    dataset.write_im(full_true, title=str(step) + " steps ground truth", 
                        filename=dataset.dataname + ".index" + str(i) + ".truth_step" + str(step), directory="test_results/truth")
        
        prev_step = step

        mean_mse = np.mean(step_mse)
        mean_mae = np.mean(step_mae)
        mean_relpred_err = np.mean(step_relpred_err)
        if has_descale:
            mean_dmae = np.mean(step_dmae)
        else:
            mean_dmae = 0
        print(step, "steps relative error:", mean_relpred_err, descale_units+":", mean_dmae, "MSE:", mean_mse, "MAE:", mean_mae)
        mse_avg.append(mean_mse)
        mae_avg.append(mean_mae)
        relpred_avg.append(mean_relpred_err)
        dmae_avg.append(mean_dmae)

        mse_min.append(np.percentile(step_mse, 5))
        mse_max.append(np.percentile(step_mse, 95))
        mae_min.append(np.percentile(step_mae, 5))
        mae_max.append(np.percentile(step_mae, 95))
        relpred_min.append(np.percentile(step_relpred_err, 5))
        relpred_max.append(np.percentile(step_relpred_err, 95))
        dmae_min.append(np.percentile(step_dmae, 5))
        dmae_max.append(np.percentile(step_dmae, 95))


    return (mse_min, mse_max, mse_avg, 
            mae_min, mae_max, mae_avg, 
            relpred_min, relpred_max, relpred_avg,
            dmae_min, dmae_max, dmae_avg)


if args.file is not None:
    # File mode
    with open(args.file, "r") as f:
        filelines = f.readlines()
    filelines = [i.strip() for i in filelines]
    filelines = [i for i in filelines if i != ""]
    lines = [i.split("\t",maxsplit=1) for i in filelines]
    if any([len(i) < 2 for i in lines]):
        lines = [re.split(r"\s{2,}", i, maxsplit=1) for i in filelines]
        if any([len(i) < 2 for i in lines]):
            print(lines)
            raise ValueError("Bad file format, see parsed lines above")
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

if not args.no_quick:
    step_arr = [1,3,5] + list(range(10,args.pred_steps+1,10))
else:
    step_arr = range(1, args.pred_steps+1)


if args.load_last:
    loaddata = np.load("test_results/lastdata.npy", allow_pickle=True)
    [mse_avgs, mse_errbounds, mae_avgs, mae_errbounds, relpred_avgs, 
        relpred_errbounds, dmae_avgs, dmae_errbounds, names] = loaddata
else:
    mse_avgs = []
    mse_errbounds = []
    mae_avgs = []
    mae_errbounds = []
    relpred_avgs = []
    relpred_errbounds = []
    dmae_avgs = []
    dmae_errbounds = []

    shape = data.shape
    data = data.reshape((1,) + shape)
    print("data shape:", data.shape)
    tfdata = tf.convert_to_tensor(data)

    for p in paths:
        m_min, m_max, m_avg, a_min, a_max, a_avg, r_min, r_max, r_avg, d_min, d_max, d_avg = run_one_test(p, data, tfdata, args.pred_steps, step_arr)
        mse_avgs.append(m_avg)
        mse_errbounds.append( (m_min, m_max) )
        mae_avgs.append(a_avg)
        mae_errbounds.append( (a_min, a_max) )
        relpred_avgs.append(r_avg)
        relpred_errbounds.append( (r_min, r_max) )
        dmae_avgs.append(d_avg)
        dmae_errbounds.append( (d_min, d_max) )
    
    savedata = [mse_avgs, mse_errbounds, mae_avgs, mae_errbounds, relpred_avgs, 
                relpred_errbounds, dmae_avgs, dmae_errbounds, names]
    np.save("test_results/lastdata.npy", savedata)

if len(mse_avgs) == 0:
    raise ValueError("No data!")

fullname = args.name + "." + dataset.dataname
if args.convolutional:
    fullname += ".conv"
fullname += "." + str(args.pred_steps)
if args.mask:
    fullname += ".masked"
if not args.no_quick:
    fullname += ".q"


def get_stats(run_avgs, index=-1, min_ind=False):
    """
    make min, avg, max from 
    """
    values = [i[index] for i in run_avgs]
    stats = {
        "min": np.min(values), 
        "avg": np.mean(values), 
        "max": np.max(values),
        "med": np.median(values),
        "std": np.std(values),
    }
    if min_ind:
        return stats, values.index(min(values))
    return stats

relpred_stats, mark = get_stats(relpred_avgs, min_ind=True)
relpred_ylim = max(1, relpred_stats["min"] * 1.2)

mse_stats = get_stats(mse_avgs)
mse_ylim = max(1, mse_stats["min"] * 1.2)

mae_stats = get_stats(mae_avgs)

print("Final relative prediction err:")
for k,v in relpred_stats.items():
    print(k + ":", v)

try:
    units = dataset.de_scale_units
except AttributeError:
    units = "n/a"

# collect stats every 10 steps
if args.no_quick:
    stats_timestep_inds = [i for i in range(len(step_arr)) if step_arr[i] % 10 == 0]
    if 0 not in stats_timestep_inds:
        stats_timestep_inds = [0] + stats_timestep_inds
else:
    stats_timestep_inds = range(len(step_arr))
with open("test_results/" + fullname + ".stats.txt", "w") as f:
    if not args.load_last:
        for p in paths:
            f.write(str(p) + "\n")
    f.write("{:<7} {:<9} {:<9} {:<9} {:<9} {:<9}\n".format("", "Min", "Avg", "Max", "Med", "Std"))
    def writeline(name, stats):
        f.write("{:<7} {min:<7.7f} {avg:<7.7f} {max:<7.7f} {med:<7.7f} {std:<7.7f}\n".format(name, **stats))
    for i in stats_timestep_inds:
        f.write("Step {}\n".format(step_arr[i]))
        writeline("RelPred", get_stats(relpred_avgs, i))
        writeline("MSE", get_stats(mse_avgs, i))
        writeline("MAE", get_stats(mae_avgs, i))
        writeline(units, get_stats(dmae_avgs, i))

# MSE Errbounds
make_plot(xrange=step_arr, data=tuple(mse_avgs), dnames=names, title="Prediction MSE -- " + args.dataset, 
    mark=mark, axlabels=("days", "mean squared error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=mse_errbounds,
    fillbetween_desc="w/ 90% confidence", ylim=mse_ylim, ymin=0,
    directory="test_results/", filename=fullname+".multistep_mse.w_confidence.png")

# MSE Clean
make_plot(xrange=step_arr, data=tuple(mse_avgs), dnames=names, title="Prediction MSE -- " + args.dataset, 
    mark=mark, axlabels=("days", "mean squared error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=None,
    fillbetween_desc="", ylim=mse_ylim, ymin=0,
    directory="test_results/", filename=fullname+".multistep_mse.png")

# RelPred Errbounds
make_plot(xrange=step_arr, data=tuple(relpred_avgs), dnames=names, title="Prediction Relative Error -- " + args.dataset, 
    mark=mark, axlabels=("days", "relative error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=relpred_errbounds,
    fillbetween_desc="w/ 90% confidence", ylim=relpred_ylim, ymin=0,
    directory="test_results/", filename=fullname+".multistep_relpred_err.w_confidence.png")

# RelPred Clean
make_plot(xrange=step_arr, data=tuple(relpred_avgs), dnames=names, title="Prediction Relative Error -- " + args.dataset, 
    mark=mark, axlabels=("days", "relative error"), legendloc="upper left",
    marker_step=(args.pred_steps // 5), fillbetweens=None,
    fillbetween_desc="", ylim=relpred_ylim, ymin=0,
    directory="test_results/", filename=fullname+".multistep_relpred_err.png")

if units != "n/a":
    # DMAE Errbounds
    make_plot(xrange=step_arr, data=tuple(dmae_avgs), dnames=names, title="Mean Absolute Error in " + units.title() + " -- " + args.dataset, 
        mark=mark, axlabels=("days", units+" MAE"), legendloc="upper left",
        marker_step=(args.pred_steps // 5), fillbetweens=dmae_errbounds,
        fillbetween_desc="w/ 90% confidence", ylim=None, ymin=0,
        directory="test_results/", filename=fullname+".multistep_mae_"+units+"_err.w_confidence.png")

    # DMAE Clean
    make_plot(xrange=step_arr, data=tuple(dmae_avgs), dnames=names, title="Mean Absolute Error in " + units.title() + " -- " + args.dataset, 
        mark=mark, axlabels=("days", units+" MAE"), legendloc="upper left",
        marker_step=(args.pred_steps // 5), fillbetweens=None,
        fillbetween_desc="", ylim=None, ymin=0,
        directory="test_results/", filename=fullname+".multistep_mae_"+units+"_err.png")


print("Results have been save to 'test_results/'")


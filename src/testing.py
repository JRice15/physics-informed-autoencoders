import os
import re
import tkinter as tk
from tkinter import filedialog

import numpy as np

from src.output_results import *


def run_name_from_weights(weights_path):
    name = re.sub(r".*/weights.", "", weights_path)
    name = re.sub(r"\.hdf5", "", name)
    return name


def run_one_test(models, weights_path, data, name, num_steps=50):
    """
    test a set of weights with multi-step prediction
    """
    autoencoder, encoder, dynamics, decoder = models

    dirname = run_name_from_weights(weights_path).strip(".")
    os.makedirs("test_results/" + dirname, exist_ok=True)

    try:
        autoencoder.load_weights(weights_path)
    except OSError as e:
        print("\n\nLooks like that run cannot be found. Make sure to enter the command line args exactly as you did during training to select those weights for testing\n\n")
        raise e

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
                    filename="pred_step" + str(step), directory="test_results/"+dirname )
                write_im(true, title=str(step) + " steps ground truth", 
                    filename="truth_step" + str(step), directory="test_results")
        
        mean_mse = np.mean(step_mse)
        print(step, "steps MSE:", mean_mse)
        error.append(mean_mse)
    
    print("")
    return error


def run_tests(args, run_name, models, data):
    """
    run test with one or two models
    """
    weights_path = "weights/weights." + run_name + "hdf5"

    # Select weights 2
    yn = input("Compare to another run? [y/n]: ")
    if yn.lower().strip() == "y":
        root = tk.Tk()
        root.withdraw()

        print("Select weights file to compare to...")
        weights2_path = filedialog.askopenfilename(initialdir="./weights/", title="select .hdf5 weights file")

        name2 = re.sub(r"\s", "_", input("Name for second chosen weights: "))

        error1 = run_one_test(models, weights_path, data, args.name)
        error2 = run_one_test(models, weights2_path, data, name2)
        dnames = (args.name, name2)
    else:
        error1 = run_one_test(models, weights_path, data, args.name)
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

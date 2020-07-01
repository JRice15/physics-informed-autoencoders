import math
import re

import cmocean
import keras
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback, History

from read_dataset import *


class ImgWriter(Callback):

    def __init__(self, model, run_name, Xtest, Ytest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.run_name = run_name
        self.X = tf.reshape(Xtest[7], (1, -1))
        write_im(Ytest[7], "Target Y(t+1)", "target_outlined", "train_results", outline=True)
        write_im(Ytest[7], "Target Y(t+1)", "target", "train_results")
        write_im(Ytest[6], "Input Y(t)", "input_outlined", "train_results", outline=True)
        write_im(Ytest[6], "Input Y(t)", "input", "train_results")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 1000 == 0:
            print("Saving result image...")
            result = self.model(self.X)
            write_im(
                im=K.eval(result),
                title="Epoch {} Prediction".format(epoch+1),
                filename=self.run_name + "__pred_epoch_{}.".format(epoch+1),
                directory="train_results"
            )


def write_im(im, title, filename, directory, show=False, outline=False):
    img = im.reshape((384, 199))

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)

    minmax = np.max(np.abs(img)) * 0.65

    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    # light contour (looks blurry otherwise)
    plt.contourf(mX, mY, img.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    # heavy contour
    if outline:
        plt.contour(mX, mY, img.T, 80, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
    im = plt.imshow(img.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)

    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
    im.axes.add_patch(p=wedge)

    plt.tight_layout()
    plt.axis('off')

    plt.title(title.title())

    filename = directory + "/" + filename + ".png"
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()


def get_num_str(num):
    if num == 0:
        return "0.0"
    if num < 0.001 or num > 999:
        num = "{:.2E}".format(num)
    else:
        num = "{:.3f}".format(num)
    return num

def save_history(H: History, run_name, marker_step=1000):
    for k in H.history.keys():
        if not k.startswith("val_"):
            train_data = H.history[k]
            try:
                valdata = H.history["val_"+k]
                mark = 1
            except KeyError:
                valdata = None
                mark = 0
            data = (train_data, valdata)
            make_plot(data=data, axlabels=("epoch",k), mark=mark,
                dnames=("train","validation"), title=k, marker_step=marker_step)
            plt.savefig("stats/" + run_name + "__" + k + ".png")
            plt.close()


def make_plot(data, title, axlabels, dnames=None, marker_step=1, 
        mark=0, legendloc="upper right"):
    """
    plot d1 and optional d2 on the same plot
    Args:
        data: tuple of lists/arrays, each of which is a data line
        mark: index of data to mark with value labels
        dnames: tuple of data names
        axlabels: tuple (xname, yname)
    """
    assert isinstance(data, tuple)

    mark_data = None
    for i in range(len(data)):
        if data[i] is None:
            continue
        if i == mark:
            mark_data = data[i]
            plt.plot(data[i], marker=".", markevery=marker_step)
        else:
            plt.plot(data[i])

    if mark_data is not None:
        points = mark_data[::marker_step]
        up = True
        for i,y in enumerate(points):
            valstr = get_num_str(y)
            if up:
                xytext = (0,5)
                up = False
            else:
                xytext = (0,-12)
                up = True
            plt.annotate(valstr, xy=(marker_step*i,y), xytext=xytext, 
                horizontalalignment="center", textcoords="offset points")
    
        valstr = get_num_str(mark_data[-1])
        ytext = 5 if up else -12
        plt.annotate(valstr, xy=(len(mark_data) - 1, mark_data[-1]), xytext=(1,ytext), textcoords="offset points")
        plt.plot(len(mark_data)-1, mark_data[-1], marker=".", color="green")

    plt.title(title)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1])
    # plt.yscale("log")
    if dnames is not None:
        dnames = [i for i in dnames if i is not None]
        plt.legend(dnames, loc=legendloc)

    plt.margins(x=0.125, y=0.1)



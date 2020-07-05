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

from src.read_dataset import *


class ImgWriter(Callback):
    """
    args:
        pipeline: tuple of (encoder, dynamics, decoder)
        freq: period of epochs to write out each image
        epochs: total epochs
    """

    def __init__(self, pipeline, run_name, Xtest, Ytest, freq=1000):
        super().__init__()
        encoder, dynamics, decoder = pipeline
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.freq = freq
        self.run_name = run_name
        self.X = tf.reshape(Xtest[7], (1, -1))
        write_im(Ytest[7], "Target Y(t+1)", "target_outlined", "train_results", outline=True)
        write_im(Ytest[7], "Target Y(t+1)", "target", "train_results")
        write_im(Ytest[6], "Input Y(t)", "input_outlined", "train_results", outline=True)
        write_im(Ytest[6], "Input Y(t)", "input", "train_results")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            print("Saving result image...")
            result = self.decoder(self.dynamics(self.encoder(self.X)))
            write_im(
                im=K.eval(result),
                title="Epoch {} Prediction".format(epoch+1),
                filename=self.run_name + "__pred_epoch_{}.".format(epoch+1),
                directory="train_results"
            )


def write_im(im, title, filename, directory, show=False, outline=False):
    """
    if show=True, filename and directory can be None
    """
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

    if show:
        plt.show()
    else:
        if filename[-1] == ".":
            ext = "png"
        else:
            ext = ".png"
        filename = directory + "/" + filename + ext
        plt.savefig(filename)

    plt.close()


def get_num_str(num):
    if num == 0:
        return "0.0"
    if num < 0.001 or num > 999:
        num = "{:.2E}".format(num)
    else:
        num = "{:.5f}".format(num)
    return num

def save_history(H: History, run_name, marker_step=1000):
    for k in H.history.keys():
        if not k.startswith("val_"):
            # skips first 200 epochs for clearer scale
            skip = 0
            if len(H.history[k]) > 200:
                skip = 200
            train_data = H.history[k][skip:]
            try:
                valdata = H.history["val_"+k][skip:]
                mark = 1
            except KeyError:
                valdata = None
                mark = 0
            data = (train_data, valdata)
            xrange = list(range(skip, len(train_data)+skip))
            make_plot(xrange=xrange, data=data, axlabels=("epoch",k), mark=mark,
                dnames=("train","validation"), title=k, marker_step=marker_step)
            plt.savefig("stats/" + run_name + "__" + k + ".png")
            plt.close()


def make_plot(xrange, data, title, axlabels, dnames=None, marker_step=1, 
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
            plt.plot(xrange, data[i], marker=".", markevery=marker_step)
        else:
            plt.plot(xrange, data[i])

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


def output_eigvals(weight_matrix, name, directory="stats", type_=None):
    try:
        weight_matrix = weight_matrix.numpy()
    except:
        weight_matrix = np.array(weight_matrix)

    e, v = np.linalg.eig(weight_matrix)

    fig = plt.figure(figsize=(6.1, 6.1), facecolor="white",  edgecolor='k', dpi=150)
    plt.scatter(e.real, e.imag, c = '#dd1c77', marker = 'o', s=15*6, zorder=2, label='Eigenvalues')

    maxeig = 1.5
    plt.xlim([-maxeig, maxeig])
    plt.ylim([-maxeig, maxeig])
    plt.locator_params(axis='x',nbins=4)
    plt.locator_params(axis='y',nbins=4)

    plt.xlabel('Real', fontsize=22)
    plt.ylabel('Imaginary', fontsize=22)
    plt.tick_params(axis='y', labelsize=22)
    plt.tick_params(axis='x', labelsize=22)
    plt.axhline(y=0,color='#636363',ls='-', lw=3, zorder=1 )
    plt.axvline(x=0,color='#636363',ls='-', lw=3, zorder=1 )

    #plt.legend(loc="upper left", fontsize=16)
    t = np.linspace(0,np.pi*2,100)
    plt.plot(np.cos(t), np.sin(t), ls='-', lw=3, c = '#636363', zorder=1 )
    plt.tight_layout()
    # plt.show()

    end = "__eigvals"
    if type_ is not None:
        end += "." + type_
    plt.savefig(directory + '/' + name + end + '.png')
    plt.close()



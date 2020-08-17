import math
import re
import os

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

    def __init__(self, pipeline, run_name, dataset: CustomDataset, freq=1000):
        super().__init__()
        encoder, dynamics, decoder = pipeline
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.freq = freq
        self.dir = "train_results/" + run_name
        self.run_name = run_name
        self.dataset = dataset
        X = tf.convert_to_tensor(dataset.Xtest[dataset.write_index])
        Y = tf.convert_to_tensor(dataset.Xtest[dataset.write_index+1])
        self.X = tf.reshape(X, (1,)+X.shape)
        self.Y = tf.reshape(Y, (1,)+Y.shape)

        self.dataset.write_im(K.eval(self.X), title="Input Y(t)", 
            filename="input."+dataset.dataname, directory="train_results")
        self.dataset.write_im(K.eval(self.Y), title="Target Y(t+1)", 
            filename="target."+dataset.dataname, directory="train_results")


    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            print("Saving result image...")
            result = self.decoder(self.dynamics(self.encoder(self.X)))
            self.dataset.write_im(
                img=K.eval(result),
                title="Epoch {} Prediction".format(epoch+1),
                subtitle=self.run_name,
                filename="pred_epoch_{}.".format(epoch+1),
                directory=self.dir,
            )


# def write_im(img, title, filename, directory, subtitle="", show=False, outline=False):
#     """
#     if show=True, filename and directory can be None
#     """
#     shape = img.shape
#     img = np.rot90(img.T, k=1)

#     x2 = np.arange(0, shape[0], 1)
#     y2 = np.arange(0, shape[1], 1)
#     mX, mY = np.meshgrid(x2, y2)

#     minmax = np.max(np.abs(img)) * 0.65

#     plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
#     # light contour (looks blurry otherwise)
#     plt.contourf(mY.T, mX.T, img, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
#     # heavy contour
#     if outline:
#         plt.contour(mY.T, mX.T, img, 40, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
#     im = plt.imshow(img, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)

#     # wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
#     # im.axes.add_patch(p=wedge)

#     plt.tight_layout()
#     # plt.axis('off')
#     plt.xticks([])
#     plt.yticks([])
#     plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
#     plt.xlabel(subtitle)

#     plt.title(title.title())

#     if show:
#         plt.show()
#     else:
#         if filename[-1] == ".":
#             ext = "png"
#         else:
#             ext = ".png"
#         filename = directory + "/" + filename + ext
#         plt.savefig(filename)

#     plt.close()


def get_num_str(num):
    if num == 0:
        return "0.0"
    if num < 0.001 or num > 999:
        num = "{:.2E}".format(num)
    else:
        num = "{:.5f}".format(num)
    return num

def save_history(H: History, run_name, secs, marker_step=1000, skip=30):
    epochs_ran = len(H.history["loss"])
    statsfile_name = "stats/" + run_name + "/stats.txt"
    with open(statsfile_name, "w") as f:
        f.write(run_name + "\n\n")
        f.write("Epochs ran:\t\t\t{}\n".format(epochs_ran))
        f.write("Secs per epoch:\t\t{}\n".format(secs / epochs_ran))
        f.write("Minutes total:\t\t{}\n".format(secs / 60))
        f.write("Hours total:\t\t{}\n".format(secs / 3600))
    for k in H.history.keys():
        if not k.startswith("val_"):
            # skips first couple epochs for clearer scale
            if len(H.history[k]) < 2 * skip:
                skip = 0
            train_data = H.history[k][skip:]
            try:
                valdata = H.history["val_"+k][skip:]
                mark = 1
            except KeyError:
                valdata = None
                mark = 0
            data = (train_data, valdata)
            xrange = list(range(skip, len(train_data)+skip))
            make_plot(xrange=xrange, data=data, axlabels=(run_name,k), mark=mark,
                dnames=("train","validation"), title=k+" by epoch", marker_step=marker_step,
                skipshift=skip, directory="stats/"+run_name, filename=k+".png")

        with open(statsfile_name, "a") as f:
            if valdata is None:
                f.write("Final {}:\t\t\t{}\n".format(k, train_data[-1]))
                f.write("Min {}:\t\t\t{}\n".format(k, min(train_data)))
            else:
                f.write("Final {}:\t\t\t{}\tval:\t{}\n".format(k, train_data[-1], valdata[-1]))
                f.write("Min {}:\t\t\t{}\tval:\t{}\n".format(k, min(train_data), min(valdata)))



def make_plot(xrange, data, title, axlabels, directory, filename, dnames=None, marker_step=1, 
        mark=0, legendloc="upper right", skipshift=0, fillbetweens=None, 
        fillbetween_desc="", ylim=None, ymin=None):
    """
    Args:
        data: tuple of lists/arrays, each of which is a data line
        mark: index of data to mark with value labels
        dnames: tuple of data names
        axlabels: tuple (xname, yname)
    """
    assert isinstance(data, tuple)

    fig, ax = plt.subplots(figsize=(6, 4.5))

    global_min = 1e20
    global_max = -1e20
    mark_data = None
    for i in range(len(data)):
        if data[i] is None:
            continue
        if i == mark:
            mark_data = data[i]
        global_min = min(np.min(data[i]), global_min)
        global_max = max(np.max(data[i]), global_max)
        plt.plot(xrange, data[i])
        if fillbetweens is not None:
            plt.fill_between(xrange, fillbetweens[i][0], fillbetweens[i][1], alpha=0.15)

    if mark_data is not None:
        mark_indices = list(range(xrange[0],xrange[-1],marker_step))
        points = np.interp(mark_indices, xrange, mark_data)
        # points = mark_data[mark_indices]
        up = True
        for i,y in enumerate(points):
            valstr = get_num_str(y)
            if up:
                xytext = (0,5)
                up = False
            else:
                xytext = (0,-12)
                up = True
            xy = (xrange[0] + marker_step*i, y)
            plt.plot(*xy, marker=".", mfc="black", mec="black", markersize=5)
            plt.annotate(valstr, xy=xy, xytext=xytext, 
                horizontalalignment="center", textcoords="offset points")
    
        valstr = get_num_str(mark_data[-1])
        ytext = 5 if up else -12
        plt.annotate(valstr, xy=(xrange[-1], mark_data[-1]), xytext=(-7,ytext), textcoords="offset points")
        plt.plot(xrange[-1], mark_data[-1], marker=".", color="green")

    plt.title(title)
    plt.xlabel(axlabels[0])
    plt.ylabel(axlabels[1] + " " + fillbetween_desc)
    plt.tight_layout()
    # plt.yscale("log")
    if dnames is not None:
        dnames = [i for i in dnames if i is not None]
        plt.legend(dnames, loc=legendloc)

    current_bot, current_top = plt.ylim()
    if ylim is not None:
        current_top = min(ylim, current_top)
    global_min = global_min - (0.03 * abs(current_top - global_min))
    bot_diff = 0
    if ymin is not None:
        global_min = min(ymin, global_min, current_bot)
        bot_diff = max(current_bot - global_min, 0)
    new_top = current_top + (0.1 * bot_diff)
    plt.ylim(bottom=global_min, top=new_top)

    plt.margins(x=0.125, y=0.1)

    fname = os.path.join(directory, filename)
    plt.savefig(fname, dpi=300)
    plt.clf()


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

    end = "eigvals"
    if type_ is not None:
        end += "." + type_
    plt.savefig(directory + '/' + name.strip(".") + "/" + end + '.png')
    plt.close()



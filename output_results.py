import math
import re

import cmocean
import keras
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from keras import backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback

from read_dataset import *


class ImgWriter(Callback):

    def __init__(self, model, run_name, Xtest, Ytest, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.run_name = run_name
        self.X = tf.reshape(Xtest[7], (1, -1))
        write_im(Ytest[7], "Target Y(t+1)", "target.", kind="truth")
        write_im(Ytest[6], "Input Y(t)", "input.", kind="truth")

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 500 == 0:
            print("Saving result image...")
            result = self.model(self.X)
            write_im(
                im=K.eval(result),
                title="Epoch {} Prediction".format(epoch+1),
                filename=self.run_name + "pred_epoch_{}.".format(epoch+1)
            )


def write_im(im, title, filename, show=False, kind="prediction"):
    img = im.reshape((384, 199))

    x2 = np.arange(0, 384, 1)
    y2 = np.arange(0, 199, 1)
    mX, mY = np.meshgrid(x2, y2)

    minmax = np.max(np.abs(img)) * 0.65

    plt.figure(facecolor="white",  edgecolor='k', figsize=(7.9,4.7))
    # light contour (looks blurry otherwise)
    plt.contourf(mX, mY, img.T, 80, cmap=cmocean.cm.balance, alpha=1, vmin=-minmax, vmax=minmax)
    # heavy contour
    if kind != "prediction":
        plt.contour(mX, mY, img.T, 80, colors='black', alpha=0.5, vmin=-minmax, vmax=minmax)
    im = plt.imshow(img.T, cmap=cmocean.cm.balance, interpolation='none', vmin=-minmax, vmax=minmax)

    wedge = mpatches.Wedge((0,99), 33, 270, 90, ec="#636363", color='#636363',lw = 5, zorder=200)
    im.axes.add_patch(p=wedge)

    plt.tight_layout()
    plt.axis('off')

    plt.title(title.title())

    filename = "results/" + filename + "png"
    plt.savefig(filename)

    if show:
        plt.show()

    plt.close()

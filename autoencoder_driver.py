import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from read_dataset import *
from regularizers import inverse_reg, lyapunov_stability_reg
from shallow_autoencoder import *

os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)

X, Xtest = data_from_name("flow_cylinder")

datashape = X[0].shape

autoencoder, encoder, dynamics, decoder = shallow_autoencoder(datashape, 1, lambda_=2)

optimizer = Adam(0.001)
autoencoder.compile(optimizer=optimizer, loss=mse)

# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, 
    min_lr=0.00005))
callbacks.append(ModelCheckpoint("weights/test.hdf5", save_best_only=True, 
    save_weights_only=True, period=10, verbose=1))

history = autoencoder.fit(
    x=X, y=Y,
    batch_size=20,
    epochs=4000,
    callbacks=callbacks,
    validation_split=0.1,
)

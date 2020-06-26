import os

import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from read_dataset import *
from regularizers import inverse_reg, lyapunov_stability_reg
from shallow_autoencoder import *

os.makedirs("data", exist_ok=True)
os.makedirs("weights", exist_ok=True)

lr = 0.001
batch_size = 34
lambda_ = 2
kappa = 3
epochs = 4000

# Read Data
X, Xtest = data_from_name("flow_cylinder")
datashape = X[0].shape

# Create Model
models = shallow_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    lambda_=lambda_,
)
autoencoder, encoder, dynamics, decoder = models

optimizer = Adam(lr)
autoencoder.compile(optimizer=optimizer, loss=mse)

# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]

callbacks = []
callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=0.00005))
callbacks.append(ModelCheckpoint("weights/test.hdf5", save_best_only=True, verbose=1))

print(X.shape, Y.shape)

history = autoencoder.fit(
    x=X, y=Y,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.1,
)

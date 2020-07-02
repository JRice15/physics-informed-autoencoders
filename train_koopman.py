import re

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard)
from keras.losses import mse
from keras.models import Model
from keras.optimizers import Adam

from src.common import *
from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3


# Initialization actions
make_dirs()
args = gather_args("koopman", num_sizes=2, defaults=Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest = data_from_name("flow_cylinder")
datashape = X[0].shape
print("datashape:", datashape)

def data_gen(X, steps):
    i = steps
    while True:
        if i+steps >= X.shape[0]:
            i = steps
            continue
        print("datagen shape:", X[i-steps:i+steps+1].shape)
        yield (X[i-steps:i+steps+1], None)

def make_val_data(Xtest, steps):
    Xs = []
    for i in range(steps+1, Xtest.shape[0]-steps):
        Xs.append(Xtest[i-steps:i+steps+1])
    return (Xs, None)


# Create Model
autoencoder = koopman_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    fwd_wt=args.forward,
    bwd_wt=args.backward,
    id_wt=args.identity,
    cons_wt=args.consistency,
    pred_steps=args.pred_steps,
    sizes=args.sizes,
)

optimizer = Adam(
    learning_rate=args.lr, 
    # clipvalue=5.0,
)
autoencoder.compile(optimizer=optimizer, loss=None)
    # experimental_run_tf_function=False)

weights_path = "weights/weights." + run_name + "hdf5"

def lr_schedule(epoch):
    """
    reduce lr by half every 1000 epochs
    """
    max_divisor = 5
    divisor = epoch // 1000
    new_rate = args.lr / (2 ** min(divisor, max_divisor))
    if epoch % 1000 == 0:
        print("LearningRateScheduler setting learning rate to {}".format(new_rate))
    return new_rate

callbacks = [
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 
    #     min_lr=(args.lr / 16), verbose=1),
    LearningRateScheduler(lr_schedule),
    ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, 
        verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    # ImgWriter(autoencoder, run_name, Xtest[:-1], Xtest[1:]),
]

print("\n\n\nBegin Training")

H = autoencoder.fit(
    x=data_gen(X, args.pred_steps),
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=make_val_data(Xtest, args.pred_steps),
    verbose=2,
)

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

save_history(H, run_name, marker_step=marker_step)

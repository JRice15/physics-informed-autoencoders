import re
import time

import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import metrics
from keras.callbacks import (LearningRateScheduler, ModelCheckpoint,
                             ReduceLROnPlateau, TensorBoard, History)
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

def data_gen(X, steps, batchsize):
    """
    yeilds lists of length batchsize, each element as 2*steps+1 consecutive snapshots
    """
    i = steps
    data = []
    while True:
        if i+steps >= X.shape[0]:
            i = steps
            continue
        if len(data) == batchsize:
            yield data
            data = []
        data.append( X[i-steps:i+steps+1] )


def make_val_data(Xtest, steps):
    Xs = []
    for i in range(steps+1, Xtest.shape[0]-steps):
        Xs.append(Xtest[i-steps:i+steps+1])
    return (Xs, None)


# Create Model
models = koopman_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    fwd_wt=args.forward,
    bwd_wt=args.backward,
    id_wt=args.identity,
    cons_wt=args.consistency,
    pred_steps=args.pred_steps,
    sizes=args.sizes,
)
autoencoder, encoder, forward_dyn, backward_dyn, decoder = models

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


H = History() # must be first for access later
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 
    #     min_lr=(args.lr / 16), verbose=1),
rate_sched = LearningRateScheduler(lr_schedule)
checkpoint = ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, 
        verbose=1, save_freq=20)
tboard = TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100)
writer = ImgWriter(model=autoencoder, run_name=run_name, Xtest=Xtest[:-1], 
        Ytest=Xtest[1:])
callbacks = [H, rate_sched, checkpoint, tboard, writer]

print("\n\n\nBegin Training")

# H = autoencoder.fit(
#     x=data_gen(X, args.pred_steps, args.batchsize),
#     y=None,
#     steps_per_epoch=3,
#     # batch_size=args.batchsize,
#     epochs=args.epochs,
#     callbacks=callbacks,
#     # validation_data=make_val_data(Xtest, args.pred_steps),
#     # validation_split=0.2,
#     # validation_batch_size=args.batchsize,
#     verbose=2,
# )


def do_callbacks(kind, logs=None, epoch=None):
    for c in callbacks:
        if epoch is not None:
            getattr(c, kind)(epoch, logs) 
        else:
            getattr(c, kind)(logs)

tboard.set_model(autoencoder)
checkpoint.set_model(autoencoder)
rate_sched.set_model(autoencoder)
writer.set_model(autoencoder)

steps_per_epoch = 3
dataset = data_gen(X, args.pred_steps, args.batchsize)

# Training Loop
do_callbacks("on_train_begin")
for epoch in range(args.epochs):
    t1 = time.time()
    do_callbacks("on_epoch_begin", epoch=epoch)

    step = 0
    for batch in dataset:
        do_callbacks("on_train_batch_begin")
        step += 1

        print("batch shape:", len(batch), "x", batch[0].shape)
        logs = autoencoder.train_on_batch(batch, return_dict=True)
        print("train logs:", logs)

        do_callbacks("on_train_batch_end", logs=logs)
        if step >= steps_per_epoch:
            break
    
    logs = autoencoder.evaluate(Xtest, batch_size=args.batchsize, return_dict=True, 
        callbacks=callbacks)
    print("eval logs:", logs)

    do_callbacks("on_epoch_end", epoch=epoch, logs=logs)

do_callbacks("on_train_end")


if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

H = callbacks[0]
save_history(H, run_name, marker_step=marker_step)

output_eigvals(forward_dyn.weights[0], run_name, type_="forward")
output_eigvals(backward_dyn.weights[0], run_name, type_="backward")

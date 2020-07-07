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
args = gather_args("koopman", num_sizes=2, defaults=Defaults)
run_name = make_run_name(args)
make_dirs(run_name)

# Read Data
X, Xtest, imshape = data_from_name(args.dataset)
datashape = X[0].shape
print("datashape:", datashape)

def format_data(X, bwd_steps, fwd_steps):
    """
    slice X into sequences of 2*steps+1 snapshots for input to koopman autoencoder
    """
    out = []
    for i in range(bwd_steps, X.shape[0]-fwd_steps):
        out.append( X[i-bwd_steps:i+fwd_steps+1] )
    return np.array(out)


# Create Model
models = koopman_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    fwd_wt=args.forward,
    bwd_wt=args.backward,
    id_wt=args.identity,
    cons_wt=args.consistency,
    forward_steps=args.fwd_steps,
    backward_steps=args.bwd_steps,
    sizes=args.sizes,
)
if args.bwd_steps > 0:
    autoencoder, encoder, forward_dyn, backward_dyn, decoder = models
else:
    autoencoder, encoder, forward_dyn, decoder = models

optimizer = Adam(
    learning_rate=args.lr, 
    # clipvalue=5.0,
)
autoencoder.compile(optimizer=optimizer, loss=None)
    # experimental_run_tf_function=False)

weights_path = "weights/weights." + run_name + "hdf5"


callbacks = [
    LearningRateScheduler(lr_schedule(args)),
    ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, 
        verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    ImgWriter(pipeline=(encoder, forward_dyn, decoder), run_name=run_name, 
        Xtest=Xtest[:-1], Ytest=Xtest[1:], freq=args.epochs//6, imshape=imshape),
]


print("\n\n\nBegin Training")

start_time = time.time()

H = autoencoder.fit(
    x=format_data(X, args.bwd_steps, args.fwd_steps),
    y=None,
    # steps_per_epoch=3,
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=(format_data(Xtest, args.bwd_steps, args.fwd_steps), None),
    # validation_split=0.2,
    # validation_batch_size=args.batchsize,
    verbose=2,
)

print("Training took {0} minutes".format((time.time() - start_time)/60))
print("{0} seconds per epoch".format((time.time() - start_time)/args.epochs))

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

save_history(H, run_name, marker_step=marker_step)

output_eigvals(forward_dyn.weights[0], run_name, type_="forward")
if args.bwd_steps > 0:
    output_eigvals(backward_dyn.weights[0], run_name, type_="backward")



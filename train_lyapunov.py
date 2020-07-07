import re
import time

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
from src.lyapunov_autoencoder import (Defaults, lyapunov_autoencoder,
                                      make_run_name)
from src.output_results import *
from src.read_dataset import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3


# Initialization actions
args = gather_args("lyapunov", num_sizes=3, defaults=Defaults)
run_name = make_run_name(args)
make_dirs(run_name)

# Read Data
X, Xtest, imshape = data_from_name(args.dataset)
datashape = X[0].shape

# targets are one timestep ahead of inputs
Y = X[:-1]
X = X[1:]
Ytest = Xtest[:-1]
Xtest = Xtest[1:]

# Create Model
models = lyapunov_autoencoder(
    snapshot_shape=datashape,
    output_dims=datashape[-1],
    kappa=args.kappa,
    lambda_=args.lambd,
    gamma=args.gamma,
    no_stability=args.no_stability,
    sizes=args.sizes,
    weight_decay=args.wd,
)
autoencoder, encoder, dynamics, decoder = models

optimizer = Adam(
    learning_rate=args.lr, 
    # clipvalue=5.0,
)
autoencoder.compile(optimizer=optimizer, loss=mse, metrics=[metrics.MeanSquaredError()])
    # experimental_run_tf_function=False)

weights_path = "weights/weights." + run_name + ".hdf5"

callbacks = [
    # ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=100, 
    #     min_lr=(args.lr / 16), verbose=1),
    LearningRateScheduler(lr_schedule(args)),
    ModelCheckpoint(weights_path, save_best_only=True, save_weights_only=True, 
        verbose=1, period=20),
    TensorBoard(histogram_freq=100, write_graph=False, write_images=True, 
        update_freq=(args.batchsize * 20), embeddings_freq=100),
    ImgWriter(pipeline=(encoder, dynamics, decoder), run_name=run_name, 
        Xtest=Xtest, Ytest=Ytest, freq=args.epochs//6, imshape=imshape),
]

print("\n\n\nBegin Training")

start_time = time.time()

H = autoencoder.fit(
    x=X, y=Y,
    batch_size=args.batchsize,
    epochs=args.epochs,
    callbacks=callbacks,
    validation_data=(Xtest, Ytest),
    verbose=2,
)

print("Training took {0} minutes".format((time.time() - start_time)/60))
print("{0} seconds per epoch".format((time.time() - start_time)/args.epochs))

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

save_history(H, run_name, marker_step=marker_step)

output_eigvals(dynamics.weights[0], run_name)

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
make_dirs()
args = gather_args("lyapunov", num_sizes=3, defaults=Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest = data_from_name("flow_cylinder")
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
    sizes=args.sizes
)
autoencoder, encoder, dynamics, decoder = models

optimizer = Adam(
    learning_rate=args.lr, 
    # clipvalue=5.0,
)
autoencoder.compile(optimizer=optimizer, loss=mse, metrics=[metrics.MeanSquaredError()])
    # experimental_run_tf_function=False)

weights_path = "weights/weights." + run_name + ".hdf5"

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
    ImgWriter(pipeline=(encoder, dynamics, decoder), run_name=run_name, 
        Xtest=Xtest, Ytest=Ytest, freq=10),
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

print("Traing took {0} minutes".format((time.time() - start_time)/60))

if 7000 >= args.epochs >= 3000:
    marker_step = 1000
else:
    marker_step = args.epochs // 6

save_history(H, run_name, marker_step=marker_step)

output_eigvals(dynamics.weights[0], run_name)

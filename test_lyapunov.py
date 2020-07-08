import argparse
import json
import os
import re
import shutil


import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model

from src.common import *
from src.lyapunov_autoencoder import *
from src.output_results import *
from src.read_dataset import *
from src.testing import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

args = gather_args("lyapunov", 3, Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest, imshape = data_from_name("flow_cylinder")
datashape = X[0].shape
data = np.concatenate([X, Xtest], axis=0)

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


run_tests(args, run_name, models, data)


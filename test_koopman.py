import argparse
import json
import os
import re
import shutil
import tkinter as tk
from tkinter import filedialog

import keras
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import Input
from keras.models import Model

from src.koopman_autoencoder import *
from src.output_results import *
from src.read_dataset import *
from src.testing import *

print("Tensorflow version:", tf.__version__) # 2.2.0
print("Keras version:", keras.__version__) # 2.4.3

args = gather_args("koopman", 2, Defaults)
run_name = make_run_name(args)

# Read Data
X, Xtest, imshape = data_from_name("flow_cylinder")
datashape = X[0].shape
data = np.concatenate([X, Xtest], axis=0)

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
    weight_decay=args.wd,
)
if args.bwd_steps > 0:
    autoencoder, encoder, dynamics, backward_dyn, decoder = models
    models = (autoencoder, encoder, dynamics, decoder)


run_tests(args, run_name, models, data)

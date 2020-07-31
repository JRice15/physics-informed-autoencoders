# Physics Informed Autoencoders


## Model Types

Physics informed:
* Koopman: Koopman consistent forward and backward dynamics
* Lyapunov: Enforces Lyapunov Stability with regularization (maybe out of data; does not support a convolutional architechture at the moment, though it very easily could)

Baselines:
* Constant: Guesses a single value for the whole field, initialized at 0
* Identity: Guesses the input


## Dependancies

I have not found a clean way to install all of the required versions of packages
on a few OS's I have tried. On OSX, what has worked for me is creating a conda
environment but using the environment's pip to install Tensorflow2.2 inside, 
as TF2.2 is not available on conda for OSX.

Strict Dependancies:
* Tensorflow 2.2, and/or Tensorflow-GPU 2.2 (I have not tried >=2.3, so it could work. <2.2 does not)
* Keras 2.4 (also haven't tried >2.4)
* (if GPU) CUDA 10.1 and CuDNN 7.5|7.6

Other Dependancies (as long as the version is compatible with the strict dependencies it should work):
* matplotlib
* cmocean
* numpy
* basemap (a deprecated matplotlib subpackage. Only required if you want to generate SST images, otherwise you can use the '--no-basemap' flag while training on an SST dataset to skip generating those images. FYI, this is the most troublesome package here. 'conda install basemap' worked on my OSX, but not windows or some linux installs. I tried building it from the [source](https://github.com/matplotlib/basemap) once but failed)
* pydot (only if the '--summary' flag is used during training)

Good luck


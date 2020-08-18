# Physics Informed Autoencoders

This is the code for a recent paper "Analyzing Koopman approaches to physics-informed machine
learninging for long-term sea-surface temperature forceasting", authored by myself,
Wenwei Xu, and Andrew August, at Pacific Northwest National Laboratory.

## What Is This

This is our research into physics-informed autoencoders for sea-surface temperature
prediction. We took as a starting point two papers by N. Benjamin Erichson and 
his collaborators: Erichson et al.'s "Physics Informed Autoencoders for Lyapunov Stable Fluid Flow
Prediction", and Azencot et al.'s "Forecasting Sequential Data Using Consistent Koopman Autoencoders". We initially developed both the Koopman and Lyapunov approaches, but found the Koopman to be better performing and more intriguing. Thus, the Lyapunov models may be a little out of date, or out of sync with the rest of the code.

We use two datasets, adapted from Erichson's works. One is computer generated vorticity of ideal fluid flow past a cylinder; the code for this dataset may be a little out of date/sync with the rest but it should work (it wasn't our main focus). The other dataset is satellite data of sea-surface temperature in the Gulf of Mexico, specifically NOAA's OI SST v2 (High Resolution) dataset. Both datasets come with a "regular" and "full" version, which is just the size of snapshots used. For SST we recommend full, but for flow cylinder we recommend regular. The SST dataset is too large to upload to github here.

## Quickstart

### Training
To train a model, do
```
python3 train_autoencoder.py --model <modeltype> --dataset <dataset> --name <name> [--convolutional] [--mask] [--seed <seed>]
```
where the arguments can be the following
* modeltype: koopman, lyapunov
* dataset: cylinder, cylinder-full, sst, sst-full
* name: unique name of your choosing
* seed: an integer. default is 0

Most hyperparameters have corresponding arguments as well. See files in the 'presets/' directory for the defaults. --save and --load can be used to save parameters to a file, or load a file with parameters from the 'presets/' directory.

Koopman consistency is specified by an argument greater than zero for '--bwd-steps'. Non-consistent Koopman methods use 0 bwd steps.

Prediction images over training will be saved in 'train_results/', while metadata on the training (graphs of loss, lr, etc over time) will be stored in 'stats/'. Training takes from 30 minutes to 2 hours on a GPU.

### Testing One Model
To test a model, do
```
python3 test_autoencoders.py --dataset <dataset> --name <name> [--convolutional]
```
This will activate a user interface to pick model .hdf5 files to load and run. The models you pick should accord with the dataset and convolutionality you chose, otherwise things will likely break.

An alternative way is to create a file with the tests you want to run, and specify it with the --file argument. See 'maintests.sh' and the testfiles it references for an example.

Results are saved to 'test_results/'. To test all ~50 seeds in 'maintest.sh' takes maybe three hours on my laptop.

### Comparing Different Models
To compare multiple different models or ensembles of models run with many seeds, use
```
python3 meta_test.py --name <name> --file <testfile>
```
This will read data from the specified 'test_results/*.stats.txt' results files of the models you want to compare. Results are saved to 'meta_results/'. Running a meta test takes just a few minutes at most.


### Putting It All Together
If you wanted to compare the performance of a convolutional vs non-convolutional method, first you would train some number of otherwise identical seeds for each method, say 5. Then, you would create a testfile for each method, listing the 5 seeds of each method in a file each. You would run the tests, which would produce two sets of results in 'test_results'. You could compare directly from there by hand, or create another testfile for a meta test, listing the two '.stats.txt' files you want to compare. Running this meta_test while create graphs comparing multiple stats (MSE, MAE, Relative Prediction Error, etc) of the two methods over time.


## Model Types

Physics informed:
* Koopman: Koopman forward (and optionally backward) dynamics
* Lyapunov: Enforces Lyapunov Stability with regularization (may be out of date; does not support a convolutional architechture at the moment, though it very easily could)

Baselines:
* Constant: Guesses a single value for the whole field, initialized at 0
* Identity: Guesses the input


## Dependancies

I have not found a clean way to install all of the required versions of packages
on a few OS's I have tried. On OSX, what has worked for me is creating a conda
environment but using the conda environment's pip to install Tensorflow2.2 inside, 
as TF2.2 is not current available via conda for OSX.

Strict Dependancies:
* Tensorflow 2.2, and/or Tensorflow-GPU 2.2 (I have not tried >=2.3, so it could work. <2.2 does not)
* Keras 2.4 (haven't tried >2.4)
* (if GPU) CUDA 10.1 and CuDNN 7.5 or 7.6

Other Dependancies (as long as the version is compatible with the strict dependencies it should work):
* matplotlib
* cmocean
* numpy
* basemap (a deprecated matplotlib subpackage. Only required if you want to generate SST images, otherwise you can use the '--no-basemap' flag while training on an SST dataset to skip generating those images. FYI, this is the most troublesome package here. 'conda install basemap' worked on my OSX, but not windows or some linux installs. I tried building it from the [source](https://github.com/matplotlib/basemap) once but failed)
* pydot and pydotplus (only if the '--summary' flag is used during training)

Good luck


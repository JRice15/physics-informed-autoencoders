#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 0:20:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /qfs/people/rice566/physics-informed-autoencoders/slurm/quicktest.out
#SBATCH -e /qfs/people/rice566/physics-informed-autoencoders/slurm/quicktest.err

module purge
module load python/anaconda3.2019.3

source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh

conda activate tf2.2

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe

python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --activation relu --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 8 16 32 --activation relu --bwd-steps 0 --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --activation relu --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 8 16 32 --bwd-steps 0 --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 16 32 64 --bwd-steps 0 --epochs 10
#python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 16 32 64 --epochs 10


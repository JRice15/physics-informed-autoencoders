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
module load python/3.6.6
module load cuda/10.0.130

source /etc/profile.d/modules.sh
source /qfs/people/rice566/tf_gpu2/bin/activate

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe

python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --activation relu --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 8 16 32 --activation relu --bwd-steps 0 --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --activation relu --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 8 16 32 --bwd-steps 0 --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 8 16 32 --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name b --convol --filters 16 32 64 --bwd-steps 0 --epochs 10
python train_autoencoder.py --model koopman --dataset sst-full --no-basemap --name k --convol --filters 16 32 64 --epochs 10


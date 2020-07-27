#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 0:20:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /qfs/people/rice566/logs/log_pia.out
#SBATCH -e /qfs/people/rice566/logs/log_pia.err

source /etc/profile.d/modules.sh
source /qfs/people/rice566/tf_gpu/bin/activate

module purge
module load cuda/9.0.176

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe

python3 train_autoencoder.py --model koopman --no-basemap --name k --convol --seed 0 --filters 8 16 32 --activation relu --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 8 16 32 --activation relu --bwd-steps 0 --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 8 16 32 --activation relu --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 8 16 32 --bwd-steps 0 --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 8 16 32 --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 16 32 64 --bwd-steps 0 --epochs 10
python3 train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 16 32 64 --epochs 10

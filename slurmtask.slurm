#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 2:00:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o log_pia.out
#SBATCH -e log_pia.err

source /etc/profile.d/modules.sh
source /qfs/people/rice566/tf_gpu/bin/activate

module purge
module load cuda/9.0.176

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders


# Tasks Go Here

python3 train_autoencoder.py --model koopman --no-basemap --name b --convol --seed 0 --filters 8 16 32 --activation relu --bwd-steps 0 


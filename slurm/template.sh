#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 2:00:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gpus:1
#SBATCH -o /qfs/people/rice566/logs/log_pia.out
#SBATCH -e /qfs/people/rice566/logs/log_pia.err

source /etc/profile.d/modules.sh
source /qfs/people/rice566/tf_gpu/bin/activate

module purge
module load cuda/9.0.176

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe


#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 0:20:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /qfs/people/rice566/physics-informed-autoencoders/slurm/logs/{logname}.out
#SBATCH -e /qfs/people/rice566/physics-informed-autoencoders/slurm/logs/{logname}.err

module purge
module load python/3.6.6
module load cuda/10.0.130

source /etc/profile.d/modules.sh
source /qfs/people/rice566/tf_gpu2/bin/activate

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe



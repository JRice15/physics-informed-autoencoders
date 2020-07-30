#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t {time}
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o /qfs/people/rice566/physics-informed-autoencoders/slurm/logs/{logname}.out
#SBATCH -e /qfs/people/rice566/physics-informed-autoencoders/slurm/logs/{logname}.err

source ~/.bash_profile

module purge
module load python/anaconda3.2019.3

source /share/apps/python/anaconda3.2019.3/etc/profile.d/conda.sh
conda activate
conda activate tf2.2

conda list
which python

ulimit -s unlimited

cd /qfs/people/rice566/physics-informed-autoencoders

# Tasks Go Below. Leave some newlines to be safe



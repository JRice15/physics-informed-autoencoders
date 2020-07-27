#!/bin/bash
#SBATCH -A br20_rice566
#SBATCH -J rice_pia
#SBATCH -t 24:00:00
#SBATCH -p all
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o log_pia.out
#SBATCH -e log_pia.err

source /etc/profile.d/modules.sh
source /qfs/people/xuwe421/hyperopt/bin/activate

module purge
module load cuda/9.0.176


ulimit -s unlimited

python3 train_autoencoder.py --model koopman --bwd-steps 0 --name b --convol --seed 0 --filters 8 16 32 --activation relu
python3 train_autoencoder.py --model koopman --name k --convol --seed 0 --filters 8 16 32 --activation relu

for s in {1..3}; do
	python3 train_autoencoder.py --model koopman --bwd-steps 0 --name b --convol --seed "$s" --filters 8 16 32 --activation relu
	python3 train_autoencoder.py --model koopman --name k --convol --seed "$s" --filters 8 16 32 --activation relu
	python3 train_autoencoder.py --model koopman --bwd-steps 0 --name b --convol --seed "$s" --filters 8 16 32 
	python3 train_autoencoder.py --model koopman --name k --convol --seed "$s" --filters 8 16 32
	python3 train_autoencoder.py --model koopman --bwd-steps 0 --name b --convol --seed "$s" --filters 16 32 64
	python3 train_autoencoder.py --model koopman --name k --convol --seed "$s" --filters 16 32 64
done


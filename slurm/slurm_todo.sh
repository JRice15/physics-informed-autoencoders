
python train_autoencoder.py --model koopman --no-basemap --name k --convol --seed 0 --filters 8 16 32 --activation relu

for s in {1..3}; do
	python train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 8 16 32 --activation relu --bwd-steps 0 
	python train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 8 16 32 --activation relu
	python train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 8 16 32 --bwd-steps 0 
	python train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 8 16 32
	python train_autoencoder.py --model koopman --no-basemap --name b --convol --seed "$s" --filters 16 32 64 --bwd-steps 0 
	python train_autoencoder.py --model koopman --no-basemap --name k --convol --seed "$s" --filters 16 32 64
done


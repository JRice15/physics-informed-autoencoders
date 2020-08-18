
### Baselines
python3 test_autoencoders.py --dataset sst-full --file testfiles/identity --name i
python3 test_autoencoders.py --dataset sst-full --file testfiles/const --name c

### Consistent Koop
python3 test_autoencoders.py --dataset sst-full --file testfiles/b-0-11 --name b
python3 test_autoencoders.py --dataset sst-full --file testfiles/k-0-11 --name k

### Conv Koop
python3 test_autoencoders.py --dataset sst-full --file testfiles/kc --name kc --convol
python3 test_autoencoders.py --dataset sst-full --file testfiles/bc --name bc --convol



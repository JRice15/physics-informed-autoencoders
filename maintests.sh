
### Baselines
python3 test_autoencoders.py --dataset sst-full --file testfiles/identity --name i --mask
python3 test_autoencoders.py --dataset sst-full --file testfiles/const --name c --mask

### Consistent Koop
python3 test_autoencoders.py --dataset sst-full --file testfiles/b-0-11 --name b --mask
python3 test_autoencoders.py --dataset sst-full --file testfiles/k-0-11 --name k --mask

### Conv Koop
python3 test_autoencoders.py --dataset sst-full --file testfiles/kc --name kc --convol --mask
python3 test_autoencoders.py --dataset sst-full --file testfiles/bc --name bc --convol --mask



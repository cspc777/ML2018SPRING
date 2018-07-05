#!/bin/bash

cd src/

if [ -f ./triplet_model_dim1500.best.hdf5 ]; then
	echo "Remove model..."
	rm -rf "./triplet_model_dim1500.best.hdf5"
fi
wget "https://www.dropbox.com/s/9hwkmbudm8gs0sa/triplet_model_dim1500.best.hdf5" 

python3 model1_test.py "../"$1

#!/bin/bash

cd src/

if [ -f ./triplet_model_dim1500.best.hdf5 ]; then
	echo "Remove ResNet50 model..."
	rm -rf "./triplet_model_dim1500.best.hdf5"
fi
wget "https://www.dropbox.com/s/9hwkmbudm8gs0sa/triplet_model_dim1500.best.hdf5" 

if [ -f ./triplet_model_IV3.best.hdf5 ]; then
	echo "Remove InceptionV3 model..."
	rm -rf "./triplet_model_IV3.best.hdf5"
fi
wget "https://www.dropbox.com/s/rkbh8vktq92l85b/triplet_model_IV3.best.hdf5"

if [ -f ./triplet_model_vgg19dim1500.best.hdf5 ]; then
	echo "Remove VGG19 model..."
	rm -rf "./triplet_model_vgg19dim1500.best.hdf5"
fi
wget "https://www.dropbox.com/s/4r9efzixxqerrzg/triplet_model_vgg19dim1500.best.hdf5" 

if [ -f ./triplet_model_XCEPT.best.hdf5 ]; then
	echo "Remove Xception model..."
	rm -rf "./triplet_model_XCEPT.best.hdf5"
fi
wget "https://www.dropbox.com/s/f440zmd2wyv4c39/triplet_model_XCEPT.best.hdf5"

python3 model2_test.py "../"$1

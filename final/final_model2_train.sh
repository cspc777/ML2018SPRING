#!/bin/bash

cd src/

python3 model1_train.py
python3 model2_IV3_train.py
python3 model2_VGG19_train.py
python3 model2_XCEPT_train.py

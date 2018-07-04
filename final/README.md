# Machine Learning Spring 2018 Final Project Whale README
Team: NTU_r06944051_學弟隊長就交給你了retry

## Package
`keras==2.0.8`
`tensorflow==1.8.0`
`numpy==1.14.5`
`pandas==0.23.0`
`sklearn==0.19.1`
`PIL==5.1.0`
`Python==3.6.5`

## Dataset download

>Before running <b><i>"download_dataset.sh"</i></b>, make sure the <b><i>"unzip"</i></b> is already installed on system.
>
>If the <b><i>unzip</i></b> isn't installed, then run:
`$ sudo apt-get install unzip` 
> 
> To download dataset, run:
`$ bash download_dataset.sh`

## Training and Predicting
> Before training and predicting, make sure that all training images reside in a directory named <b><i>"train"</i></b>, all testing images reside in a directory named <b><i>"test"</i></b> and that <b><i>"train.csv"</i></b> and the two directories reside in the same directory as that of the source codes.
> The directory structure should look like the following:
> 
> ```
>    ./
>    +-- train/
>    |   +-- 00a29f63.jpg
>    |           .
>    |           .
>    |           .
>    +-- test/
>    |   +-- 000bc353.jpg
>    |           .
>    |           .
>    |           .
>    +-- train.csv
>    +-- model1_train.py
>    +-- model1_test.py
>    |           .
>    |           .
>    |           .
>    +-- final_model1_train.sh
>    +-- final_model1_test.sh
>    |           .
>    |           .
>    |           .
> ```
> Also note that for training and testing images, the cropped version of them must be used to achieve the target performance.
---
### Commands and details for model 1 
To train model 1, run:
`$ bash final_model1_train.sh`
 
To test model 1, run:
`$ bash final_model1_test.sh [PREDICTION_FILE_NAME]`
where `[PREDICTION_FILE_NAME]` is the intended name of the prediction csv.
> Note that the testing script removes existing model and download the uploaded model before predicting; therefore, to have the script use a model trained by TA, one need to comment out the "rm" command in "final_model1_test.sh"
---
### Commands and details for model 2
To train model 2, run:
`$ bash final_model2_train.sh`

To test model 2, run:
`$ bash final_model2_test.sh [PREDICTION_FILE_NAME]`
where `[PREDICTION_FILE_NAME]` is the intended name of the prediction csv.
> Note that the testing script removes existing models and download the uploaded models before predicting; therefore, to have the script use models trained by TA, one need to comment out the "rm" commands in "final_model2_test.sh"

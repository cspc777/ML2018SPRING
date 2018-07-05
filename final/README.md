# Machine Learning Spring 2018 Final Project Whale README
Team: NTU_r06944051_學弟隊長就交給你了retry
<mark>Feel free to contact us if any problems are encountered while reproducing the results!!!</mark>
> Note that all script execution commands listed in this README should be run in the same directory as Report.pdf!
## Package Version
`keras==2.0.8`
`tensorflow==1.8.0`
`numpy==1.14.5`
`pandas==0.23.0`
`sklearn==0.19.1`
`PIL==5.1.0`
`h5py==2.8.0`
`Python==3.6.5`

## Dataset download

>Before running "download_dataset.sh", make sure the <b><i>"unzip"</i></b> is already installed on system.
>
>If <b><i>"unzip"</i></b> isn't installed, run:
`$ sudo apt-get install unzip` 
> 
To download the dataset, run:
`$ bash download_dataset.sh`

## Training and Predicting
> Before training and predicting, make sure that all training images reside in a directory named <b><i>"train"</i></b>, all testing images reside in a directory named <b><i>"test"</i></b> and that <b><i>"train.csv"</i></b> and the two directories reside in the same directory (i.e. the "src" directory) as that of the python source codes.
> The directory structure should look like the following after running "download_dataset.sh":
> 
> ```
>    ./
>    +-- src/    
>    |   +-- train/
>    |   |   +-- 00a29f63.jpg
>    |   |   +       .
>    |   |   +       .
>    |   |   +       .
>    |   |
>    |   +-- test/
>    |   |   +-- 000bc353.jpg
>    |   |   +       .
>    |   |   +       .
>    |   |   +       .
>    |   |
>    |   +-- train.csv
>    |   +-- model1_train.py
>    |   +-- model1_test.py
>    |   +           .
>    |   +           .
>    |   +           .
>    |
>    |
>    +-- Report.pdf
>    +-- README.md
>    +-- final_model1_train.sh
>    +-- final_model1_test.sh
>    +           .
>    +           .
>    +           .
> ```
> Note that the shell scripts reside in the same directory as Report.pdf, as mentioned at the start of README. The shell script exexution commands should be run at this directory, too!  
> Also note that for training and testing images, the cropped version of them (the ones downloaded via "download_dataset.sh") must be used to achieve the target performance.
---
### Commands and details for model 1 
To train model 1, run:
`$ bash final_model1_train.sh`
> Note that the newly trained model will be stored in the "src" directory


To test model 1, run:
`$ bash final_model1_test.sh [PREDICTION_FILE_NAME]`
where `[PREDICTION_FILE_NAME]` is the intended name of the prediction csv.
> Note that the testing script removes existing model and download the uploaded model before predicting; therefore, to have the script use a model trained by TA, one will need to comment out the "rm" command in "final_model1_test.sh"
---
### Commands and details for model 2
To train model 2, run:
`$ bash final_model2_train.sh`
> Note that the newly trained models will be stored in the "src" directory


To test model 2, run:
`$ bash final_model2_test.sh [PREDICTION_FILE_NAME]`
where `[PREDICTION_FILE_NAME]` is the intended name of the prediction csv.
> Note that the testing script removes existing models and download the uploaded models before predicting; therefore, to have the script use models trained by TA, one will need to comment out the "rm" commands in "final_model2_test.sh"
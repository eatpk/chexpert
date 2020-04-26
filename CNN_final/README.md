# Big Data Health - Team 6 - Chest Xray Disease Diagnosis

## Intro

ReadMe document for our project Assignment

### Python 3.6 Libraries 

We utilized the following python Libraries

```
keras==2.2.4
pandas==0.23.4
h5py==2.8.0
sklearn==0.20.1
opencv==3.4.3
numpy==1.15.4
tensorflow==1.15.0
pyspark==2.4.5
```

### Code

train.py = Main file that creates, configures, and trains the NN models

ROC_Callback.py = File that prcoesses the overall accuracy for the model

Preprocess.py = File used to parse through the images for preprocessing

generator.py = File used to create the dataset for the NN models

/dataset_list/ = the .csv containing image paths to train/validation/testing set and labels for each disease.


### Execute code

1. Configure the environment and code in train.py
2. run the code via 
```
python train.py
```



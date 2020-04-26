# Big Data Health - Team 6 - Chest Xray Disease Diagnosis

## Intro

ReadMe document for our project Assignment

### Python 3.6 Libraries 

We utilized the following python Libraries

```
h5py==2.8.0
pandas==0.23.4
numpy==1.15.4
keras==2.2.4
sklearn==0.20.1
opencv==3.4.3
tensorflow==1.15.0
pyspark==2.4.5
```

### Training

1. Download this code folder.
2. Download  "dataset_list" folder.
3. Download and extract images from https://nihcc.app.box.com/v/ChestXray-NIHCC. Put all images in one folder.
4. In train.py, select model name "VGG16" or "DenseNet121". 
5. In train.py, change "image_csv_list_loc" to "dataset_list" folder in step 2. 
6. In train.py, change "image_file_loc" to image folder in step 3. 
7. Train the model. (Make sure output folder is empty before training.)
8. Training results are saved in 'output' folder.



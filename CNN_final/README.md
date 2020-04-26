# CNN Training and Testing

## Introduction

These document shows how to train CNN model for Chest X-Ray image disease diagnosis, and how to test on single images or batches.

### Prerequisites

You need the following packages in python 3.6. Please install if you don't have.

GPU is necessary for training.

```
h5py==2.8.0
pandas==0.23.4
numpy==1.15.4
keras==2.2.4
sklearn==0.20.1
opencv==3.4.3
tensorflow-gpu==1.12.0
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

### Testing Single Image

1. Please put you testing image in folder "test_images". We already provided three images here.
2. Download model weights from https://drive.google.com/drive/folders/1c550T4HrJCmD96LHx4txJau7jtWWdhSm?usp=sharing. Put them in the 'CNN_final' folder. Do NOT rename weights.
2. In test_on_single_image.py, select model name "VGG16" or "DenseNet121". 
3. In test_on_single_image.py, change "image_name" on line 19 to your target image name.
4. In test_on_single_image.py, change "true_labels_loc" on line 23 to "dataset_list" folder in step 2 in Training section above. This will provide true label if exists in our testing set.
5. Run test_on_single_image.py.

### Testing On Batch

1. In test_on_batch.py, select model name "VGG16" or "DenseNet121". 
2. In test_on_batch.py, change "image_csv_list_loc" on line 2 to "dataset_list" folder in step 2 in Training section above.
3. In test_on_batch.py, change "image_file_loc" to image folder in step 3 in Training section above.
4. Run test_on_batch.py.

import pandas as pd
import numpy as np
import os
import sys
import keras

from sklearn.metrics import roc_auc_score 
from keras.callbacks import Callback
from datetime import datetime
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications import  ResNet50
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from my_generator import DataGenerator
from ROC_Callback import ROCCallback


############################################################################################################
# EXTRACT TO SEPERATE FILE???? 


class_weight = [{0: 0.096035456, 1: 0.903964544}, 
                {0: 0.019804505, 1: 0.980195495}, 
                {0: 0.100710339, 1: 0.899289661}, 
                {0: 0.159832433, 1: 0.840167567}, 
                {0: 0.046214559, 1: 0.953785441}, 
                {0: 0.054629349, 1: 0.945370651}, 
                {0: 0.010078319, 1: 0.989921681}, 
                {0: 0.030502095, 1: 0.969497905}, 
                {0: 0.033161314, 1: 0.966838686}, 
                {0: 0.016076741, 1: 0.983923259}, 
                {0: 0.016501730, 1: 0.983498270}, 
                {0: 0.014303928, 1: 0.985696072}, 
                {0: 0.025851497, 1: 0.974148503}, 
                {0: 0.001675672, 1: 0.998324328}]
############################################################################################################






############################################################################################################
# Initialization ###########################################################################################
############################################################################################################

# Select which model to train
model_name = "DenseNet121" 
# model_name = "VGG16" 
# model_name = "res50" 


# Directory Locations for data
GCP_bucket_loc = "https://storage.googleapis.com/bigteam-team6/images/"
results_loc = "./results/" + model_name
image_csv_loc = "./dataset_list"
train_csv_loc = os.path.join(image_csv_loc, "train.csv")
val_csv_loc = os.path.join(image_csv_loc, "validation.csv")
AUC_save_path = os.path.join(results_loc, "validation_auc_log.txt") 

#import CSV for Train and Validation Data
train_csv = pd.read_csv(train_csv_loc)
val_csv = pd.read_csv(val_csv_loc)

#diseases tracked
disease_name = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]


# Initialize Model specific vars
if model_name == "VGG16":
    initial_LR = 0.0001 
    epochs = 10
    batch_size = 32
if model_name == "res50":
    initial_LR = 0.0001 
    epochs = 3
    batch_size = 32
if model_name == "DenseNet121":
    initial_LR = 0.001
    epochs = 10
    batch_size = 32

# Create Results Directory
os.system("sudo mkdir "+ results_loc) 
############################################################################################################



############################################################################################################
# Generate DataSets ########################################################################################
############################################################################################################

train_DataGenerator = DataGenerator(image_file_loc = GCP_bucket_loc, labels = train_csv, batch_size = batch_size, disease_name = disease_name, shuffle = True, target_shape = (224, 224))
validation_DataGenerator = DataGenerator(image_file_loc = GCP_bucket_loc, labels = val_csv, batch_size = batch_size, disease_name = disease_name, shuffle = False, target_shape = (224, 224))
############################################################################################################




############################################################################################################
# Build model ##############################################################################################
############################################################################################################

input_tensor = Input(shape=(224, 224, 3))

if model_name == "VGG16":
    base_model = VGG16(include_top=False, weights="imagenet", pooling="avg")
    y = base_model(input_tensor)
    y = Dense(len(disease_name), activation="sigmoid")(y)
    model = Model(inputs=input_tensor, outputs=y)

if model_name == "res50":
    base_model = ResNet50(include_top = False, weights='imagenet')
    y = base_model(input_tensor)
    y = GlobalAveragePooling2D()(y)
    y = Dense(256, activation='relu')(y)
    y = Dense(len(disease_name), activation='sigmoid')(y)
    model = Model(inputs=input_tensor, outputs=y)

if model_name == "DenseNet121":
    base_model = DenseNet121(include_top=False, weights="imagenet", pooling="avg") 
    y = base_model(input_tensor)
    y = Dense(len(disease_name), activation="sigmoid")(y)
    model = Model(inputs=input_tensor, outputs=y)
############################################################################################################





############################################################################################################
# End of Training functions ################################################################################
############################################################################################################

model.compile(optimizer=Adam(lr=initial_LR), loss="binary_crossentropy")

# Call Back Functions
saveMetrics = TensorBoard(log_dir=os.path.join(results_loc, "TensorBoard"), batch_size=batch_size)

calculate_AUC = ROCCallback(disease_name=disease_name,log_loc = AUC_save_path, validation_data=validation_DataGenerator )

reduce_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode="min", min_lr=1e-8) 

create_checkpoint = ModelCheckpoint(os.path.join(results_loc,save_weights_only=True,monitor='val_loss', "weights.{epoch:02d}-{val_loss:.2f}.h5"))
############################################################################################################



############################################################################################################
# Train Model ##############################################################################################
############################################################################################################
model.fit_generator(generator=train_DataGenerator, class_weight=class_weight,workers=1,epochs=epochs, steps_per_epoch=len(train_file_list) // batch_size,validation_data=validation_sequence,validation_steps=len(val_file_list) // batch_size, shuffle=False,callbacks=[create_checkpoint, reduce_learningRate, saveMetrics, calculate_AUC])
############################################################################################################
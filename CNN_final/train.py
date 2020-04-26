import pandas as pd
import numpy as np
import os
import sys
import keras
import pyspark

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
from generator import DataGenerator
from ROC_Callback import ROCCallback





############################################################################################################
# Initialization ###########################################################################################
############################################################################################################

# Select which model to train
model_name = "DenseNet121" 
# model_name = "VGG16" 
# model_name = "Res50" 


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

def class_weights(path):
    
    sc=pyspark.SparkContext()
    rdd = sc.textFile(path).map(lambda line: line.split(","))
    disease_counts=rdd.map(lambda x : [1 if x[i]=='1' else 0 for i in range(3,17)]).reduce(lambda x,y: [sum(i) for i in zip(x,y)])

    return [{0:elm/rdd.count(),1:1-elm/rdd.count()} for elm in disease_counts]

    
# Initialize Model specific vars
if model_name == "VGG16":
    initial_LR = 0.0001 
    epochs = 10
    batch_size = 32
if model_name == "Res50":
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

if model_name == "Res50":
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
tensorboard = TensorBoard(log_dir=os.path.join(results_loc, "tf_board"), batch_size=batch_size)

calculate_AUC = ROCCallback(disease_name=disease_name,log_loc = AUC_save_path, validation_data=validation_DataGenerator )

reduce_learningRate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode="min", min_lr=1e-8) 

create_checkpoint = ModelCheckpoint(os.path.join(results_loc,save_weights_only=True,monitor='val_loss', "weights.{epoch:02d}-{val_loss:.2f}.h5"))
############################################################################################################



############################################################################################################
# Train Model ##############################################################################################
############################################################################################################
class_weight=class_weights(train_file_list_path)

model.fit_generator(generator=train_DataGenerator, class_weight=class_weight,workers=1,epochs=epochs, steps_per_epoch=len(train_file_list) // batch_size,validation_data=validation_sequence,validation_steps=len(val_file_list) // batch_size, shuffle=False,callbacks=[create_checkpoint, reduce_learningRate, tensorboard, calculate_AUC])
############################################################################################################

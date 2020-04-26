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
from my_generator import DataGenerator


############################################################################################################
# EXTRACT TO SEPERATE FILE???? 
disease_name = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

def class_weights(path):
    
    sc=pyspark.SparkContext()
    rdd = sc.textFile(path).map(lambda line: line.split(","))
    disease_counts=rdd.map(lambda x : [1 if x[i]=='1' else 0 for i in range(3,17)]).reduce(lambda x,y: [sum(i) for i in zip(x,y)])

    return [{0:elm/rdd.count(),1:1-elm/rdd.count()} for elm in disease_counts]



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
# callbacks  ###############################################################################################
############################################################################################################
# Reference: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

class roc_callback(Callback):
    def __init__(self, validation_data, disease_name, log_loc):
        self.validation_data = validation_data
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.disease_name = disease_name
        self.log_loc = log_loc

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.validation_data)
        y_true = self.validation_data.getTrueLabel()

        aucList = []
        for i in range(len(self.disease_name)):
            one_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucList.append(one_auc)
        
        print("Mean AUC: {}\n".format(np.mean(aucList)))
        f = open(self.log_loc, "a+")
        f.write("Epoch {}: {}\n".format(epoch+1, np.mean(aucList))) # epoch starts from 0, add 1 to keep consistency.
        f.close()
        return

model.compile(optimizer=Adam(lr=initial_LR), loss="binary_crossentropy")



compute_AUC = roc_callback(validation_data=validation_DataGenerator, disease_name=disease_name,log_loc = AUC_save_path)

checkpoint = ModelCheckpoint(os.path.join(results_loc, "weights.{epoch:02d}-{val_loss:.2f}.h5"),save_weights_only=True,monitor='val_loss')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, mode="min", min_lr=1e-8)        

tensorboard = TensorBoard(log_dir=os.path.join(results_loc, "TensorBoard"), batch_size=batch_size)
############################################################################################################



############################################################################################################
# Train Model ##############################################################################################
############################################################################################################
class_weight=class_weights(train_file_list_path)

model.fit_generator(generator=train_DataGenerator, class_weight=class_weight,workers=1,epochs=epochs, steps_per_epoch=len(train_file_list) // batch_size,validation_data=validation_sequence,validation_steps=len(val_file_list) // batch_size, shuffle=False,callbacks=[checkpoint, reduce_lr, tensorboard, compute_AUC])
############################################################################################################

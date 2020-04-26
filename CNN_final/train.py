import pandas as pd
import numpy as np
import os
import sys
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
from my_generator import DataGenerator
from sklearn.metrics import roc_auc_score 
from keras.callbacks import Callback
from datetime import datetime

# 0. Set Hyper Parameters.
###################################################################################################
model_name = "DenseNet121" # must be "VGG16" or "DenseNet121"

image_csv_list_loc = "./dataset_list"
image_file_loc = "https://storage.googleapis.com/bigteam-team6/images/"
output_loc = "./output/" + model_name
disease_name = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
               "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]

epochs = 10
batch_size = 32

if model_name == "DenseNet121":
    initial_LR = 0.001

# class_weight are pre_calculated based on the training dataset. Same order as disease_name.
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

os.system("sudo mkdir "+output_loc)# create output folder
###################################################################################################


# 1. Import Data
###################################################################################################
# load data split file names.
train_file_list_path = os.path.join(image_csv_list_loc, "train.csv")
train_file_list = pd.read_csv(train_file_list_path)

val_file_list_path = os.path.join(image_csv_list_loc, "validation.csv")
val_file_list = pd.read_csv(val_file_list_path)

# generators
train_sequence = DataGenerator(image_file_loc = image_file_loc, disease_name = disease_name, labels = train_file_list, batch_size = batch_size,
                               target_shape = (224, 224), shuffle = True)

validation_sequence = DataGenerator(image_file_loc = image_file_loc, disease_name = disease_name, labels = val_file_list, batch_size = batch_size,
                               target_shape = (224, 224), shuffle = False)

print("Loading Data Done\n")
###################################################################################################


# 2. Build Model 
###################################################################################################
input_tensor = Input(shape=(224, 224, 3))


if model_name == "DenseNet121":
    base_model = DenseNet121(include_top=False, weights="imagenet", pooling="avg") 

x = base_model(input_tensor)
predictions = Dense(len(disease_name), activation="sigmoid")(x)
model = Model(inputs=input_tensor, outputs=predictions)

print(model.summary())
print("Building Model Done\n")
###################################################################################################


# 3. Callbacks
###################################################################################################
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

AUC_save_path = os.path.join(output_loc, "validation_auc_log.txt") # used to select the best model by AUC.

compute_AUC = roc_callback(validation_data=validation_sequence, 
                           disease_name=disease_name,
                           log_loc = AUC_save_path)

checkpoint = ModelCheckpoint(os.path.join(output_loc, "weights.{epoch:02d}-{val_loss:.2f}.h5"),
                             save_weights_only=True,
                             monitor='val_loss')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, 
                              patience=1, # use patience = 1, otherwise, training costs too much time.
                              mode="min", 
                              min_lr=1e-8)        

tensorboard = TensorBoard(log_dir=os.path.join(output_loc, "TensorBoard"), 
                          batch_size=batch_size)
###################################################################################################


# 4. Training
###################################################################################################
model.fit_generator(generator=train_sequence,
                              steps_per_epoch=len(train_file_list) // batch_size,
                              epochs=epochs,
                              validation_data=validation_sequence,
                              validation_steps=len(val_file_list) // batch_size,
                              callbacks=[checkpoint, reduce_lr, tensorboard, compute_AUC],
                              class_weight=class_weight,
                              workers=1, 
                              shuffle=False)
###################################################################################################
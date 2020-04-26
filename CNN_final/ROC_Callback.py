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





# Reference: https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras

class ROCCallback(Callback):
    
    def __init__(self, validation_data, disease_name, log_loc):
        self.validation_data = validation_data
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.disease_name = disease_name
        self.log_loc = log_loc

    def on_epoch_end(self, epoch, logs={}):
        y_true_value = self.validation_data.getTrueLabel()
        y_prediction = self.model.predict_generator(self.validation_data)
        

        accuracyList = []

        for i in range(len(self.disease_name)):
            disease_acc = roc_auc_score(y_true_value[:, i], y_prediction[:, i])
            accuracyList.append(disease_acc)
        
        mean_acc = np.mean(accuracyList)
        
        logfile = open(self.log_loc, "a+")
        logfile.write("epoch {}: {}\n".format(epoch, mean_acc)) 
        logfile.close()
        
        return
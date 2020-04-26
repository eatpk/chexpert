import numpy as np
import pandas as pd
import keras

import os
import cv2
import requests

# reference: A detailed example of how to use data generators with Keras
# https://stanford.edu/~shervine/blog

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_file_loc, disease_name, labels, batch, target_shape=(224,224), shuffle=True):
        'Initialization'
        self.image_file_loc = image_file_loc
        self.target_shape = target_shape
        self.shuffle = shuffle
        self.labels = labels
        self.disease_name = disease_name
        self.batch = batch
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.labels) / self.batch))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch:(index+1)*self.batch]
        batched_set = self.labels.iloc[indexes] 

        X, y = self.data_generator(batched_set)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.labels))
        if self.shuffle :
            np.random.shuffle(self.indexes)

    def image_processor(self,path):
        resp = requests.get(path,stream=True).raw
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image,-1)
        image = cv2.resize(image, self.target_shape, interpolation = cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        image = image.astype(np.float64)
        image /= 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        return image

    def data_generator(self, batched_set):

        paths = [os.path.join(self.image_file_loc, x) for x in batched_set.iloc[:,0].tolist()]
        X = np.array([self.image_procrocessor(x) for x in paths])
        y = batched_set[self.disease_name].values

        return X, y

import numpy as np
import pandas as pd
import keras
#from keras.preprocessing.image import ImageDataGenerator
import os
import cv2

# Our generator is built based on the blog in the reference below.
# reference: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_file_loc, disease_name, labels, batch_size=32, target_shape=(224,224), shuffle=True):
        'Initialization'
        self.image_file_loc = image_file_loc
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.labels = labels
        self.disease_name = disease_name
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        batchLabels = self.labels.iloc[indexes] # use indexes to find corresponding rows.

        # Generate data
        X, y = self.__data_generation(batchLabels)

        return X, y

    # this is necessary for validation and testing
    def getTrueLabel(self):
        return self.labels.iloc[:int(np.floor(len(self.labels) / self.batch_size)) * self.batch_size][self.disease_name].values

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batchLabels):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # function to read image to np array. 
        def readImg(path):
            resp = requests.get(path,stream=True).raw
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image=cv2.imdecode(image,-1)
            image = cv2.resize(image, self.target_shape, interpolation = cv2.INTER_CUBIC)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Note: opencv read BGR rather than RGB, need convert here.
            image = image.astype(np.float64) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

            return image

        # Initialization
        imagePathList = batchLabels["Image Name"].tolist()
        imagePathList = [os.path.join(self.image_file_loc, x) for x in imagePathList]

        X = np.array([readImg(x) for x in imagePathList])
        y = batchLabels[self.disease_name].values

        return X, y

import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

PATH_TRAIN = "CheXpert-v1.0-small/train.csv"
PATH_VALID = "CheXpert-v1.0-small/valid.csv"

print('===> Loading entire datasets')
print(PATH_TRAIN)
train_data = pd.read_csv(PATH_TRAIN)
valid_data = pd.read_csv(PATH_VALID)

#Many images are not labels in training dataset for Pneumonia columns,so
#removing those images from the dataset. there are three labels- 0,1,-1
train_data=train_data[train_data.Pneumonia.notnull()].reset_index()

#keeping image size as 224,224 across the board.
new_size=(224,224)

image_tensor=torch.empty(224, 224)
image_tensor=image_tensor.unsqueeze(0)

#traing preporcessing for first 100 images

target=torch.from_numpy(np.array(train_data['Pneumonia'][0:99])).type(torch.long)

for i in range(0,len(train_data['Path'][0:99])):
    
    image_path=train_data['Path'][i]
    img = Image.open(image_path).convert('L')
    img=img.resize(new_size)
    img0 = torch.from_numpy(np.asarray(img)).type(torch.float32)
    img0 /= 255.0
    temp_torch = img0.unsqueeze(0)
    image_tensor = torch.cat((temp_torch, temp_torch), dim=0)

#data set will be a tuple of tensors
#first tensor - (number of images, height, width)
#second tensor - (labels of the images)
    
dataset = (image_tensor, target)
print(dataset)
BATCH_SIZE=20
NUM_WORKERS=8
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)



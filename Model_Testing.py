
# coding: utf-8

# # Model Testing

# In[30]:


# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import logging

plt.ion()   # interactive mode


# In[31]:


# SETTINGS
FREEZE_WEIGHTS = True
BATCH_SIZE = 10
NUM_CLASSES = 25
NUM_EPOCHS = 1

# Local Directory
#data_dir = 'processed_data'

# Directory on DAS4
data_dir = '/var/scratch/prai1809/processed_data1'


# In[32]:


# The CNN we use wants input dimensions of 3x224x224
#
# We use dataset of 2D faces where width x height is not 224x224
#  -> Rescale to 256 as first transformation


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'alternate_test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Data folder structure:
# /augmented_dataset
#   - /train
#       - /class_1
#       - /class_2
#   - /val
#       - /class_1
#       - /class_2

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test', 'alternate_test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test', 'alternate_test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test', 'alternate_test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[33]:


def test_model(model, criterion, optimizer):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    print()
    print('Evaluating Test Set')
    logging.info('Evaluating Test Set')
    
    model.eval()   # Set model to evaluate mode

    phase = 'alternate_test'
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
    
    loss = running_loss / dataset_sizes[phase]
    acc = running_corrects.double() / dataset_sizes[phase]
    
    print('Test Acc: {:4f}'.format(acc))
    logging.info('Test Acc: {:4f}'.format(acc))
    


# In[34]:


# Load model
model_fileName = "FREEZEWEIGHTS_False_CLASSES_25_BATCHSIZE_10_EPOCHS_2_TRAIN_10500_VAL_2250_Test2250"
model_t1 = torch.load(model_fileName)


# In[35]:


#model_conv = model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_t1.fc.parameters(), lr=0.001, momentum=0.9)


# In[36]:


test_model(model_t1, criterion, optimizer_conv)


# In[37]:


#get_ipython().system('jupyter nbconvert --to script Model_Testing.ipynb')


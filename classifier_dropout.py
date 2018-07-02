
# coding: utf-8

# ### Model Settings

# In[32]:


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

import argparse
from torch.autograd import Variable
from cnn_finetune import make_model

plt.ion()   # interactive mode


# ## Settings

# In[33]:


BATCH_SIZE = 10
NUM_CLASSES = 25
NUM_EPOCHS = 5
DROPOUT_RATE = 0.5

# Local Directory
#data_dir = 'processed_data'

# Directory on DAS4
data_dir = '/var/scratch/prai1809/star_train'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ## Convert Notebook to Python Script to use on server
# Comment out line below before running on server. 

# In[34]:


#get_ipython().system('jupyter nbconvert --to script classifier_dropout.ipynb')


# ## Load Data

# In[35]:


transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

train_set = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)

val_set = datasets.ImageFolder(root=os.path.join(data_dir, "val"), transform=transform)
val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

test_set = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)

classes = train_set.classes


# ### Logging

# In[36]:


m_filename = 'DROPOUT_{}_FROZEN_{}_CLASSES_{}_BATCHSIZE_{}_EPOCHS_{}_TRAIN_{}_VAL_{}_TEST_{}'.format(DROPOUT_RATE,
                                                                                          False, 
                                                                                          NUM_CLASSES, 
                                                                                          BATCH_SIZE,NUM_EPOCHS, 
                                                                                          len(train_set.classes), 
                                                                                          len(train_set), 
                                                                                          len(val_set),
                                                                                          len(test_set))

LOG_FILENAME = m_filename + ".log"
logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)
logging.info(LOG_FILENAME)

print(LOG_FILENAME)


# ## Create Model

# In[37]:


model = make_model(
    'resnet18',
    pretrained=True,
    num_classes=len(classes),
    dropout_p=DROPOUT_RATE,
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# ### Train function

# In[38]:


def train(epoch):
    total_loss = 0
    total_size = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        total_loss += loss.item()
        total_size += data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))
            
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAverage loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss / total_size))


# ### Test Function

# In[39]:


def test(phase):
    model.eval()
    test_loss = 0
    correct = 0
    
    loader = test_loader
    
    if phase == "val":
        loader = val_loader
    
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum().item()

    
            
    test_loss /= len(loader.dataset)
    print('\nTest set: Average loss: {:4f}, Accuracy: {}/{} ({:4f}%)\n'.format(
        test_loss, correct, 
        len(loader.dataset),
        100. * correct / len(loader.dataset)))
    
    logging.info('\n{} set: Average loss: {:4f}, Accuracy: {}/{} ({:4f}%)\n'.format(phase,
        test_loss, correct, 
        len(loader.dataset),
        100. * correct / len(loader.dataset)))


# ### Train Model

# In[40]:


since = time.time()

for epoch in range(0, NUM_EPOCHS):
    print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
    logging.info('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
    train(epoch)
    test('val')

test('test')
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
logging.info('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))


# ### Save Model

# In[ ]:


torch.save(model, m_filename)


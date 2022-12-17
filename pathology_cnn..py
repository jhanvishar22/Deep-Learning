#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

import torchvision
from torchvision import datasets, transforms, models

import torchvision.transforms as t
from torchvision import datasets




torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)


# In[20]:


datadirec = "/Users/22jha/OneDrive/Desktop/pathologyData/train"
testdirec = "/Users/22jha/OneDrive/Desktop/pathologyData/test"


# In[21]:


trainset = ImageFolder(datadirec,transform = transforms.Compose([
    transforms.Resize((100,100)),transforms.ToTensor()
]))
testset = ImageFolder(testdirec,transforms.Compose([
    transforms.Resize((100,100)),transforms.ToTensor()
]))


# In[22]:


trainset


testset


# In[23]:


image, labels = trainset[0]
print(image.shape,labels)


# In[24]:


print("Classes of dataset : \n",trainset.classes)


# In[25]:


batchsize = 128


traindl = DataLoader(trainset, batchsize, shuffle = True, num_workers = 4)
testdl = DataLoader(testset, batchsize, num_workers = 4)


# In[26]:


class ImageClassificationBase(nn.Module):

    def training_step(self, batch):
        images, labels = batch
        # Generating predictions and loss function
        out = self(images)     
        lossdl = F.cross_entropy(out, labels) 
        return lossdl

    def validation_step(self, batch):
        images, labels = batch
        # Generating predictions and loss and accuracy functions
        out = self(images)
        lossdl = F.cross_entropy(out, labels)
        accur = accuracy(out, labels)
        
        return {'val_loss': lossdl.detach(), 'val_acc': accur}

    def validation_epoch_end(self, outputs):
        b_loss = [x['val_loss'] for x in outputs]
        # Combining losses and accuracy
        eloss = torch.stack(b_loss).mean()   
        bacc = [x['val_acc'] for x in outputs]
        eacc = torch.stack(bacc).mean()      
        return {'val_loss': eloss.item(), 'val_acc': eacc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))


# In[27]:


class CNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size = 3, padding = 1,stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(16, 8, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Flatten(start_dim=1),

            nn.Linear(25*25*8,64),
            nn.Linear(64,2),
            nn.ReLU()
        )

    def forward(self, xb):
        return self.network(xb)


# In[28]:


def train(model, data_loader, optimizer, criterion, epoch):
    model.train()
    trainloss = 0
    ncorr = 0
    for idx, (data, target) in enumerate(data_loader):
        data, target = data, target
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        trainloss += loss.item()
        prediction = output.argmax(dim=1)
        ncorr += prediction.eq(target).sum().item()
        if idx % 50 == 0:
            print('Train Epochs: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\tAccuracy: {:.0f}%'.format(
                epoch, idx * len(data), len(data_loader.dataset),
                100. * idx / len(data_loader), trainloss / (idx + 1),
                100. * ncorr / (len(data) * (idx + 1))))
    trainloss /= len(data_loader)
    traccuracy = ncorr / len(data_loader.dataset)
    return trainloss, traccuracy


# In[29]:


def test(model, data_loader, criterion):
    model.eval()
    testloss = 0
    tscor = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data, target
            output = model(data)
            loss = criterion(output, target)
            testloss += loss.item()  # sum up batch loss
            prediction = output.argmax(dim=1)
            tscor += prediction.eq(target).sum().item()
    testloss /= len(data_loader)
    tsaccuracy = tscor / len(data_loader.dataset)
    return testloss, tsaccuracy


# In[30]:


modelts = CNN()
criteriats = nn.CrossEntropyLoss()
optimizerts = torch.optim.Adam(modelts.parameters(), lr=0.001)


# In[31]:


for epoch in range(1, 6):
    lstrain, actrain = train(modelts, traindl, optimizerts, criteriats, epoch)
    print('Epoch {} Train: Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(
        epoch, lstrain, 100. * actrain))


# In[18]:


lstest, actest = test(modelts, testdl, criteriats)
print('Test : Loss: {:.4f}, Accuracy: {:.3f}%\n'.format(lstest, 100. * actest))


# In[ ]:





# In[ ]:





# In[ ]:





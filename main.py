# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:24:39 2022

@author: tripa
"""


'''
Acknowledgement: https://programmer.group/a-simple-classification-of-pascalvoc-data-set-with-pytorch.html
'''

import os
import copy
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils import VOCClassification

import torch
from  torch import optim, nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tensorboard')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 4
learning_rate = 0.005
n_epochs = 30

# %%

# Prepare VOC07 classification data

root_dir = './'
dict_path = 'VOC2007/ImageSets/Main/'
categories = [file[:-10] for file in os.listdir(dict_path) if '_train.txt' in file]

train_img = []
val_img = []
test_img = []

# For images containing multiple objects, the image is included multiple times with each label present in the image
for file in os.listdir(dict_path):
    if '_train.txt' in file:
        fo = open(dict_path + file)
        cat = categories.index(file[:-10])
        l = [(line[:-4], cat) for line in iter(fo) if int(line[-3:]) == 1]
        train_img.extend(l)
    elif '_test.txt' in file:
        fo = open(dict_path + file)
        cat = categories.index(file[:-9])
        l = [(line[:-4], cat) for line in iter(fo) if int(line[-3:]) == 1]
        test_img.extend(l)
    elif '_val.txt' in file:
        fo = open(dict_path + file)
        cat = categories.index(file[:-8])
        l = [(line[:-4], cat) for line in iter(fo) if int(line[-3:]) == 1]
        val_img.extend(l)

train_img.sort()
val_img.sort()
test_img.sort()

print(len(train_img), len(val_img), len(test_img))

# %%

train_transforms = transforms.Compose([transforms.ToTensor(),
                                       transforms.Resize((224, 224)),
                                       transforms.RandomResizedCrop(224), 
                                       transforms.RandomHorizontalFlip(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((224, 224)),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

trainset = VOCClassification(root_dir, train_img, train_transforms)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

valset = VOCClassification(root_dir, val_img, test_transforms)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = VOCClassification(root_dir, test_img, test_transforms)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

# %%

# Load pre-trained resnet
net = models.resnet18(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, 20)  # add linear layer with 20 output nodes
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
multistep_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,25], gamma=0.1)

train_acc_history = []
train_loss_history = []
val_acc_history = []
val_loss_history = []

best_model_wts = copy.deepcopy(net.state_dict())
best_acc = 0.0

for epoch in range(1, n_epochs+1):
    
    # Variables to record time taken in each epoch
    since = time.time()
    
    print('Epoch {}/{}'.format(epoch, n_epochs))
    print('-' * 10)

    # Each epoch will have a train and a test phase
    for phase in ['train', 'val']:
        
        if phase == 'train':
            net.train()  # Set model to training mode
        else:
            net.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0
        
        if phase == 'train':
            dataloader = train_loader
            n_samples = trainset.__len__()
        else:
            dataloader = val_loader
            n_samples = valset.__len__()

        for images, labels in tqdm(dataloader, desc='Batch'):
            
            images, labels = images.to(device), labels.to(device)
            
            # Make gradients zero before forward pass
            optimizer.zero_grad()
        
            # Set to True in training phase
            with torch.set_grad_enabled(phase == 'train'):
            
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                
                loss = criterion(outputs, labels)
        
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(predictions == labels)
        
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        
        if phase == 'train':
            train_acc_history.append(epoch_acc)
            train_loss_history.append(epoch_loss)
            writer.add_scalar("train_loss", epoch_loss, epoch)
            writer.add_scalar("train_acc", epoch_acc, epoch)
        
        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(net.state_dict())
        if phase == 'val':
            val_acc_history.append(epoch_acc)
            val_loss_history.append(epoch_loss)
            writer.add_scalar("val_loss", epoch_loss, epoch)
            writer.add_scalar("val_acc", epoch_acc, epoch)
        
    time_elapsed = time.time() - since
    print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # load best model weights
    net.load_state_dict(best_model_wts)
    
    # Save the best model
    torch.save(net.state_dict(), 'voc_resnet.pth')
    
# After the training, print best validation accuracy
print('Best val Acc: {:4f}'.format(best_acc))

writer.flush()
writer.close()

# %%

# Plot loss and accuracy curves
plt.plot(train_loss_history, label = 'Train Loss')
plt.plot(val_loss_history, label = 'Val Loss')
plt.legend()
plt.show()

plt.plot(train_acc_history, label = 'Train Accuracy')
plt.plot(val_acc_history, label = 'Val Accuracy')
plt.legend()
plt.show()

# %%

# Test the model

running_loss = 0
running_corrects = 0

for images, labels in tqdm(test_loader, desc='Batches'):
    
    images, labels = images.to(device), labels.to(device)
    
    outputs = net(images)
    _, predictions = torch.max(outputs, 1)
    
    loss = criterion(outputs, labels)
    
    running_loss += loss.item() * batch_size
    running_corrects += torch.sum(predictions == labels)
    
test_loss = running_loss / trainset.__len__()
test_acc = running_corrects.double() / trainset.__len__()

print("Test Loss: {:.4f}, Test Acc: {:.4f}".format(test_loss, test_acc))
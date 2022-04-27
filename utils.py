# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 11:24:31 2022

@author: tripa
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset
        
class VOCClassification(Dataset):
    
    def __init__(self, root_dir, image_set, transforms=None):
        
        self.root_dir = root_dir
        self.transforms = transforms
        
        self.images = list()
        self.labels = list()
        for x in image_set:
            self.images.append(x[0])
            self.labels.append(x[1])
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (image, target) where target is the label of the image
            """
            image_path = os.path.join(self.root_dir, 'VOC2007', 'JPEGImages', self.images[index]+'.jpg')
            img = Image.open(image_path).convert("RGB")

            if self.transforms is not None:
                img = self.transforms(img)
            
            target = torch.tensor(self.labels[index])
            
            return img, target
    
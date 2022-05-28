import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as tf
import os


class SmokeDataset(Dataset):
    def __init__(self, img_list, label_list, train=True):
        self.train_transform = tf.Compose([tf.Resize((224,224)),
                                    tf.RandomChoice([
                                    tf.RandomHorizontalFlip(),
                                    tf.RandomRotation(degrees = (-45,45)), 
                                    tf.ColorJitter(0.2, 0.2, 0.2, 0.2),
                                    ]),
                                    tf.ToTensor(), 
                                    tf.Normalize((0.4810, 0.4591, 0.4134),(0.2522, 0.2258, 0.2291))])
        self.valid_transform = tf.Compose([tf.Resize((224,224)),
                                    tf.ToTensor(), 
                                    tf.Normalize((0.4810, 0.4591, 0.4134),(0.2522, 0.2258, 0.2291))])
        self.img = img_list
        self.label = label_list
        self.train = train
    def __len__(self):
        return len(self.img)
    def __getitem__(self, index):
        image = Image.open(self.img[index]).convert('RGB')
        if self.train : 
            image = self.train_transform(image)
        else:
            image = self.valid_transform(image)
        return image, torch.tensor(self.label[index], dtype=torch.long)


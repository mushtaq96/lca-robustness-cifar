# -*- coding: utf-8 -*-
"""
Created on Sat May  6 13:49:19 2023

@author: jakob
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
# mapping of traffic sign classes
from utils.class_mapping import sign_map

# To-Do: Implement function for calculating adversarial perturbation mask

"""
----------------------------------------------------------------------------------------------------------------
train_transforms
transforms training data by using different measures such as augmentation, resizing, normalizing, ...
---------------------------
inputs:
    resize_to           --> size of resized image
    rot_range           --> range of rotation in degrees
    mean                --> mean of image normalization
    std                 --> standard deviation of image normalization
---------------------------
output:
    transforms_train    --> collection of training image transformations
----------------------------------------------------------------------------------------------------------------
"""
def train_transforms(resize_to, rot_range=None, mean=None, std=None):
    transforms_train = transforms.Compose([
        transforms.Resize(size = resize_to),
        transforms.ToTensor(),
        #transforms.RandomRotation(rot_range),
        #transforms.ColorJitter(brightness=0.2)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms_train


"""
----------------------------------------------------------------------------------------------------------------
val_transforms
transforms validation data by using different measures such as augmentation, resizing, normalizing, ...
---------------------------
inputs:
    resize_to           --> size of resized image
    mean                --> mean of image normalization
    std                 --> standard deviation of image normalization
---------------------------
output:
    transforms_train    --> collection of validation image transformations
----------------------------------------------------------------------------------------------------------------
"""
def val_transforms(resize_to, mean=None, std=None):
    transforms_val = transforms.Compose([
        transforms.Resize(size = resize_to), 
        transforms.ToTensor(),
        #transforms.Normalize(mean = mean, 
        #                     std = std)
        ])
    return transforms_val


"""
----------------------------------------------------------------------------------------------------------------
adv_transforms
transforms data for adversarial learning
---------------------------
inputs:
    resize_to           --> size of resized image
---------------------------
output:
    transforms_train    --> collection of validation image transformations
----------------------------------------------------------------------------------------------------------------
"""
def adv_transforms(resize_to):
    transforms_val = transforms.Compose([
        transforms.Resize(size = resize_to), 
        transforms.ToTensor(),
        ])
    return transforms_val


"""
----------------------------------------------------------------------------------------------------------------
dataset_sanity_check
plots images of a dataset to check whether transformations and ordering happened correctly
---------------------------
inputs:
    dataset             --> torch dataset consisting of images and labels
---------------------------
output:
                        --> figure with plots
----------------------------------------------------------------------------------------------------------------
"""
def dataset_sanity_check(dataset):
    # create figure
    figure = plt.figure(figsize=(8,8))
    ax = figure.add_subplot(1, 1, 1)
    cols, rows = 3, 3
    
    # plot images with labels as titles
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size = (1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(sign_map[label])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0), cmap="gray")
        
    return 


"""
----------------------------------------------------------------------------------------------------------------
np_dataset_sanity_check
plots images of a dataset to check whether transformations and ordering happened correctly
---------------------------
inputs:
    np_images             --> numpy array consisting of images
    np_labels             --> numpy array consisting of corresponding labels
---------------------------
output:
                        --> figure with plots
----------------------------------------------------------------------------------------------------------------
"""
def np_dataset_sanity_check(np_images, np_labels):

    # Visualise some of the training data
    figure = plt.figure(figsize=(10,10))
    ax = figure.add_subplot(1, 1, 1)
    cols, rows = 3, 3
    
    for i in range(1,10):
        figure.add_subplot(rows, cols, i)
        plt.title(sign_map[int(np_labels[i])])
        plt.imshow(np.moveaxis(np_images[i], [0, 1, 2], [2, 0, 1]))
        
    return figure



"""
----------------------------------------------------------------------------------------------------------------
batch_sanity_check
plots images of dataloader batch to check whether batching has worked correctly
---------------------------
inputs:
    dataloader          --> torch dataloader consisting of images and labels
---------------------------
output:
                        --> figure with plot
----------------------------------------------------------------------------------------------------------------
"""
def batch_sanity_check(dataloader):
    # read images and labels
    batch_images, batch_labels = next(iter(dataloader))
    
    # create figure and plot image
    figure = plt.figure(figsize=(8,8))
    ax = figure.add_subplot(1, 1, 1)
    
    plt.title(sign_map[batch_labels[0].item()])
    plt.imshow(batch_images[0].permute(1, 2, 0), cmap="gray")
    
    return figure


"""
----------------------------------------------------------------------------------------------------------------
vis_class_dist
creates an image of the class distribution among training and validation dataset
---------------------------
inputs:
    train_data          --> torch training dataset consisting of images and labels
    val_data            --> torch validation dataset consisting of images and labels
---------------------------
output:
                        --> figure with plot
----------------------------------------------------------------------------------------------------------------
"""
def vis_class_dist(train_data, val_data, img_path):
    # initialize list of zeros for all different sign classes
    train_sign_dist = [0] * len(sign_map)
    
    # iterate over training and validation dataset and increment class counts
    for image in range(len(train_data)):
        train_sign_dist[train_data[image][1]] +=1
    
    val_sign_dist = [0] * len(sign_map)
    for image in range(len(val_data)):
        val_sign_dist[val_data[image][1]] +=1    
    
    # initialize figure and provide plot relevant data
    figure = plt.figure(figsize = (70,35))
    ax = figure.add_subplot(1, 1, 1)
    
    label_list = []
    for label in sign_map.values():
        label_list.append(label)
    ax.set_xticklabels(label_list)
    ax.xaxis.set_ticks(np.arange(0, len(label_list), 1.0))
    plt.xticks(rotation=90, fontsize=25)
    plt.yticks(fontsize=25)
    plt.bar(range(len(sign_map)), train_sign_dist, label="Training", color="green")
    plt.bar(range(len(sign_map)), val_sign_dist, label="Validation", color="dodgerblue")
    legend = plt.legend(loc = 'upper right', fontsize=30)
    #plt.title("Traffic Sign Distribution", fontsize=25)
    plt.xlabel("Klasse Verkehrszeichen", fontsize=35)
    plt.ylabel("Anzahl Vorkommnisse", fontsize=35)
    
    # insert images of classes
    for label in range(len(sign_map)):
        image = Image.open(img_path + f"\\{label}\\" + "img0.png")
        imagebox = OffsetImage(image, zoom=2.0)
        imagebox.image.axes = ax
        
        ax.add_artist(
            AnnotationBbox(imagebox, (label*1.0, train_sign_dist[label]+30))
            )
        
    return figure, train_sign_dist, val_sign_dist


"""
----------------------------------------------------------------------------------------------------------------
vis_class_dist
creates an image of the class distribution among training and validation dataset
---------------------------
inputs:
    train_data          --> list or array of training labels
---------------------------
output:
                        --> figure of distribution
----------------------------------------------------------------------------------------------------------------
"""
def vis_sub_class_dist(train_labels):
    # initialize list of zeros for all different sign classes
    train_sign_dist = [0] * len(sign_map)
    
    # iterate over label array and increment class counts
    for image in range(len(train_labels)):
        train_sign_dist[int(train_labels[image])] +=1
        
    # initialize figure and provide plot relevant data
    figure = plt.figure(figsize = (10,5))
    ax = figure.add_subplot(1, 1, 1)
    plt.bar(range(len(sign_map)), train_sign_dist, label="train")
    legend = plt.legend(loc = 'upper right')
    plt.title("Traffic Sign Distribution")
    plt.xlabel("Traffic Sign ID")
    plt.ylabel("Numer of occurences")
    
    return figure

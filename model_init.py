# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:12:48 2023

@author: jakob
"""
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from train import *
"""
----------------------------------------------------------------------------------------------------------------
initialize_model
load an available model architecture from torch and adjust the architecture for the specific problem
---------------------------
inputs:
    num_classes         --> number of classes for the classification task
    feature_extract     --> flag, whether gradients shall be frozen --> True for finetuning
    weights             --> weights, which shall be used for initialization
---------------------------
output:
    model_ft            --> model after problem specific adjustment
    input_size          --> input size of the model 
----------------------------------------------------------------------------------------------------------------
"""

import os



def initialize_model(model_name, num_classes, feature_extract, weights, pretrained):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    
    cwd = os.getcwd()
    print(f"The current working directory is {cwd}.")

    if model_name == 'resnet18':
        resnetPretrainedPath = '/home/bokhars/thesis/robustness/resnet18_pretrained.pth'
        if pretrained == True:
            model_ft = models.resnet18(weights=None)
            model_ft = torch.load(resnetPretrainedPath) # Load Resnet18 with pretrained weights (on imagenet)
        else:
            model_ft = models.resnet18(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == 'resnet50':
        resnetPretrainedPath = '/home/bokhars/thesis/robustness/resnet50_pretrained.pth'
        if pretrained == True:
            model_ft = models.resnet50(weights=None)
            model_ft = torch.load(resnetPretrainedPath) # Load Resnet50 with pretrained weights (on imagenet)
        else:
            model_ft = models.resnet50(weights=weights)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224       
    elif model_name == 'alexnet':
        alexPretrainedPath = '/home/bokhars/thesis/robustness/alexnet_pretrained.pth'
        if pretrained == True:
            model_ft = models.alexnet(weights=None)
            model_ft = torch.load(alexPretrainedPath) # Load Alexnet with pretrained weights (on imagenet)
        else:
            model_ft = models.alexnet(weights=weights)
        in_features = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(in_features, num_classes)

    print("Loaded model from file:", model_ft)

    return model_ft, input_size

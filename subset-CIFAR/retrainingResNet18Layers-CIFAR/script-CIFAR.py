# -*- coding: utf-8 -*-
# %%
from __future__ import print_function
from __future__ import division
import random
import torch
import numpy as np
# setting the seed value for reproducibiity
# Set the random seed for PyTorch
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True # deterministic behavior for cuDNN to make GPU computations reproducible
import sys
sys.path.append('/home/bokhars/thesis/robustness/')

import torch.nn as nn
import torch.optim as optim
from IPython.display import Image, display
from dataloading import *
from utils.class_mapping import stem_dir 
from model_init import *
from analysis.eval_model import *
from train import *
from lcapt.lca import LCAConv2D
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
from utils.layer_reset import decide_reset, decide_unfreeze

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv1D, LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error
from lcapt.preproc import make_unit_var, make_zero_mean


  
""" Check for availability of GPU """
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


""" Inputs """
# Model
model_name = "LCA Frontend + ResNet18"                # model architecture which shall be used
num_classes = 10                                    # number of classes to be differentiated
# Training
batch_size = 128                                     # size of training and validation batches
num_epochs = 5
lr = 0.1                                          # learning rate ResNet
eta = 0.01                                          # learning rate lca layer 
weight_decay = 0.0002

FEATURES = 64                                       # number of dictionary features to learn
KERNEL_SIZE = 7                                    # height and width of each kernel
LAMBDA = 0.5                                        # LCA threshold for inference
LCA_ITERS = 100
print_freq = 1
STRIDE = 2                                          # convolutional stride
TAU = 300                                           # LCA time constant

# paths
base_model_dir = stem_dir + "/models/_params/subset-CIFAR/retrainingResNet18Layers/" 
result_dir = stem_dir + "/models/_params/subset-CIFAR/retrainingResNet18Layers/" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"

fig_path = stem_dir + "/figures/subset-CIFAR/retrainingResNet18Layers/Reset/"
fig_path_inference = stem_dir + "/figures/subset-CIFAR/retrainingResNet18Layers/Reset/Inference_" +  f"_lr_{lr}_bs_{batch_size}" + ".png"

os.makedirs(os.path.dirname(base_model_dir), exist_ok=True)
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_inference), exist_ok=True)

LCA_model_path = stem_dir + "/models/_params/subset-CIFAR/" + "lca_dict" + f"_lr_0.001_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}_iters_{LCA_ITERS}_tau_{TAU}" + ".pt"
backbone_CIFAR_path = stem_dir + "/models/_params/subset-CIFAR/backbone_CIFAR/" + f"backbone_CIFAR_lr_{lr}_bs_{batch_size}" + ".pt"

""" Define model """
class LCA_Frontend_Basemodel_TinyImageNet(nn.Module):
    def __init__(self, lca_model, resnet18, num_classes):
        super().__init__()
        self.LCA = lca_model
        self.decoder = resnet18
        self.decoder.conv1 = nn.Conv2d(FEATURES, 64, 7, 2, 3) # (input channels, output channels, kernel size, padding) Why FEATURES should it not be 3?
        self.decoder.fc = nn.Linear(512, num_classes) 


    def forward(self, x):
        #  Forward pass through the LCA layer
        inputs, code, x1, recon_error = self.LCA(x)
        # Forward pass through the modified ResNet18 decoder
        x = self.decoder(code)
        return x
        
def get_unfrozen_layers(model):
    unfrozen_layers = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            unfrozen_layers.append(name)
    return unfrozen_layers
# %%
if __name__ == '__main__':
    
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    class_dict = {}
    for i in range(len(categories)):
        class_dict[i] = categories[i]

    """ Load Data """
    path = '/home/bokhars/thesis/robustness/architectures/subset-CIFAR'
    train_dataset = torch.load(path + '/cifar10-train.pth')
    test_dataset = torch.load(path + '/cifar10-test.pth')

    # sanity checking
    # count the number of images in each class in the training subset
    train_class_counts = [0] * 10  # 3 is number of classes in subset
    for i in range(len(train_dataset)):
        label = train_dataset[i][1]
        train_class_counts[label] += 1

    # Count the number of images in each class in the testing subset
    test_class_counts = [0] * 10
    for i in range(len(test_dataset)):
        label = test_dataset[i][1]
        test_class_counts[label] += 1

    # Print the class counts for the training and testing subsets
    print('Training subset class counts:', train_class_counts)
    print('Testing subset class counts:', test_class_counts)
    layers_to_unfreeze = []
    layer1 = ['layer1.0.conv1.weight', 'layer1.0.bn1.weight', 'layer1.0.bn1.bias',
                'layer1.0.conv2.weight', 'layer1.0.bn2.weight', 'layer1.0.bn2.bias',
                'layer1.1.conv1.weight', 'layer1.1.bn1.weight', 'layer1.1.bn1.bias',
                'layer1.1.conv2.weight', 'layer1.1.bn2.weight', 'layer1.1.bn2.bias']
    layer2 = ['layer2.0.conv1.weight', 'layer2.0.bn1.weight', 'layer2.0.bn1.bias',
                'layer2.0.conv2.weight', 'layer2.0.bn2.weight', 'layer2.0.bn2.bias',
                'layer2.1.conv1.weight', 'layer2.1.bn1.weight', 'layer2.1.bn1.bias',
                'layer2.1.conv2.weight', 'layer2.1.bn2.weight', 'layer2.1.bn2.bias']
    layer3 = ['layer3.0.conv1.weight', 'layer3.0.bn1.weight', 'layer3.0.bn1.bias',
                'layer3.0.conv2.weight', 'layer3.0.bn2.weight', 'layer3.0.bn2.bias',
                'layer3.1.conv1.weight', 'layer3.1.bn1.weight', 'layer3.1.bn1.bias',
                'layer3.1.conv2.weight', 'layer3.1.bn2.weight', 'layer3.1.bn2.bias']
    layer4 = ['layer4.0.conv1.weight', 'layer4.0.bn1.weight', 'layer4.0.bn1.bias',
                'layer4.0.conv2.weight', 'layer4.0.bn2.weight', 'layer4.0.bn2.bias',
                'layer4.1.conv1.weight', 'layer4.1.bn1.weight', 'layer4.1.bn1.bias',
                'layer4.1.conv2.weight', 'layer4.1.bn2.weight', 'layer4.1.bn2.bias']           
    # Train the model with different sets of unfrozen layers
    for i in range(0, 5):
        # Train the model with different sets of unfrozen layers
        layers_to_unfreeze = ['conv1.weight', 'bn1.weight', 'bn1.bias']
        if i == 1:
            # Unfreeze layer1 and its batch normalization layers
            layers_to_unfreeze.extend(layer1)
        elif i == 2:
            # Unfreeze layer2 and its batch normalization layers
            layers_to_unfreeze.extend(layer1)
            layers_to_unfreeze.extend(layer2)
        elif i == 3:
            # Unfreeze layer3 and its batch normalization layers
            layers_to_unfreeze.extend(layer1)
            layers_to_unfreeze.extend(layer2)
            layers_to_unfreeze.extend(layer3)
        elif i == 4:
            # Unfreeze layer4 and its batch normalization layers
            layers_to_unfreeze.extend(layer1)
            layers_to_unfreeze.extend(layer2)
            layers_to_unfreeze.extend(layer3)
            layers_to_unfreeze.extend(layer4)

        layers_to_unfreeze += ['fc.weight', 'fc.bias']
        # Print the layers to unfreeze for this iteration
        print(f"Iteration {i}: Layers to unfreeze - {layers_to_unfreeze}")
        # Create training and validation dataloaders
        # Create a DataLoader with worker_init_fn and generator
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
                  
        """ Load LCA model """
        lca_pretrained = LCAConv2D(
                out_neurons=FEATURES,
                in_neurons=3,
                result_dir=result_dir,
                kernel_size=KERNEL_SIZE,
                stride=STRIDE,
                lambda_=LAMBDA,
                eta=eta,
                tau=TAU,
                lca_iters=LCA_ITERS,
                pad="valid",
                track_metrics=False,
                return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
                transfer_func="hard_threshold",
            )
        
        lca_pretrained.load_state_dict(torch.load(LCA_model_path))
        
        weight_grid = make_feature_grid(lca_pretrained.get_weights())
        plt.imshow(weight_grid.float().cpu().numpy())
        
        """ Load Resnet18 model """
        resnet18 = models.resnet18(weights=None)
        # print(resnet18)
        
        num_ftrs = resnet18.fc.in_features
        resnet18.fc = nn.Linear(num_ftrs, num_classes)
        
        resnet18.load_state_dict(torch.load(backbone_CIFAR_path))
    
        updated_resnet18 = decide_reset(resnet18, layers_to_unfreeze)
      
        model = LCA_Frontend_Basemodel_TinyImageNet(lca_model=lca_pretrained, resnet18=updated_resnet18, num_classes=num_classes)
        model = model.to(device)
            
        # filer used because only the unfrozen parameters are to be updated during training
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay) 
    
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
            
        model_dir = base_model_dir + f"Layer{i}" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".pt"
        fig_path_acc = fig_path + f"Layer{i}_Acc_over_epochs" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png" 
        fig_path_loss = fig_path + f"Layer{i}_Loss_over_epochs" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png" 
        os.makedirs(os.path.dirname(fig_path_acc), exist_ok=True)
        os.makedirs(os.path.dirname(fig_path_loss), exist_ok=True)
        
        """ Train Model """
        _, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = train_model(model, device, train_loader, test_loader, 
                                                                                    criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler)
                                                                                                                    
        # Plot training and model performance
        train_params=[lr, weight_decay, batch_size, num_epochs]
        lca_params = [None]
        plot_model_perf(model_name, train_params, lca_params, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, num_epochs=num_epochs, 
                        fig_path_acc=fig_path_acc, fig_path_loss=fig_path_loss)
        
        """ Inference of model"""
        plot_inference_results_custom_model(test_loader, model, class_dict, device=device, fig_path=fig_path_inference)
        

        """ Save model """
        torch.save(model.state_dict(), model_dir)
        if i == 0:
            key = f"L {i}"
        else:
            key = f"L 0-{i}"
        best_val_accuracy = max(val_acc_hist)
        value = "{:.5f}".format(best_val_accuracy)

        filePath = 'robustness/architectures/subset-CIFAR/retrainingResNet18Layers-CIFAR/'
        try:
            with open(filePath + 'overall_accuracies_unfreezing_resetting.txt', 'a') as file:
                file.write(f"{key}, {value}\n")
        except Exception as e:
            print(f"Error writing to file: {e}")

# %%


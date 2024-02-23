# -*- coding: utf-8 -*-
# This is CIFAR-Backbone, trained with CIFAR-10, from scratch on ResNet18
# %%
import random
import sys
sys.path.append('/home/bokhars/thesis/robustness/')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
from dataloading import *
from utils.class_mapping import stem_dir 
from model_init import *
from analysis.eval_model import *
from train import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from torch.utils.data import Dataset
from torch.optim import lr_scheduler


# setting the seed value for reproducibiity
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True # deterministic behavior for cuDNN to make GPU computations reproducible
torch.backends.cudnn.benchmark = False # disable cuDNN benchmarking, which can introduce variability in GPU computations

""" Inputs """
data_dir = '/net/smtcac0060/fs2/scr/scr2/datasets/cifar/' #'/net/smtcac2623/fs1/scr/scr1/mushtaq/cifar'
# Model
model_name = "Backbone CIFAR"                            # model architecture which shall be used
num_classes = 10                                    # number of classes to be differentiated
# Training
batch_size = 128                                     # size of training and validation batches
num_epochs = 50                                    # num of epochs for training
lr = 0.1                                          # learning rate used for training
weight_decay = 0.0002                          

model_dir = stem_dir + "/models/_params/subset-CIFAR/backbone_CIFAR/" + f"backbone_CIFAR_lr_{lr}_bs_{batch_size}" + ".pt"

fig_path_cm = stem_dir + "/figures/subset-CIFAR/backbone_CIFAR/conf_matrix_" + f"basemodel_lr_{lr}_bs_{batch_size}" +  ".png"
fig_path_acc = stem_dir + "/figures/subset-CIFAR/backbone_CIFAR/Acc_over_epochs_" + f"basemodel_lr_{lr}_bs_{batch_size}" + ".png"
fig_path_loss = stem_dir +"/figures/subset-CIFAR/backbone_CIFAR/Loss_over_epochs_" + f"basemodel_lr_{lr}_bs_{batch_size}" + ".png"
fig_path_inference = stem_dir + "/figures/subset-CIFAR/backbone_CIFAR/Inference_" +  f"basemodel_lr_{lr}_bs_{batch_size}" + ".png"

os.makedirs(os.path.dirname(model_dir), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_acc), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_loss), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_inference), exist_ok=True)

if __name__ == '__main__':
    
    """ Check for availability of GPU """
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    
    train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    class_dict = {}
    for i in range(len(categories)):
        class_dict[i] = categories[i]

    """ Load Data """
    train_dataset = torchvision.datasets.CIFAR10(root= data_dir, train = True, download = False, transform = train)
    test_dataset = torchvision.datasets.CIFAR10(root= data_dir, train = False, download = False, transform = test)
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
    
    # Create training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
    
    """ Train model from scratch """
    # Initialize the non-pretrained version of the model used for this run
    model,_ = initialize_model('resnet18', num_classes, feature_extract=False, weights=None, pretrained=False)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) 
 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    _, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = train_model(model, device, train_loader, test_loader, 
                                                                                  criterion, optimizer, num_epochs=num_epochs, scheduler=scheduler)
    
    # Plot training and model performance
    train_params=[lr, weight_decay, batch_size, num_epochs]
    lca_params = [None]
    plot_model_perf(model_name, train_params, lca_params, train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, num_epochs=num_epochs, 
                    fig_path_acc=fig_path_acc, fig_path_loss=fig_path_loss)
    
     
    """ Inference of scratch model"""
    plot_inference_results(test_loader, model, class_dict, device=device, fig_path=fig_path_inference)
    

    """ Save model """
    torch.save(model.state_dict(), model_dir)
    
    
    """ Confusion matrix for validation data """
    # plt_confusion_matrix(val_dataloader, model, "torch", device, batch_size, fig_path_cm)    
# %%

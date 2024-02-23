# -*- coding: utf-8 -*-
# This script creates LCA dictionary for CIFAR-10 subset dataset
# %%
import random
import sys
sys.path.append('/home/bokhars/thesis/robustness/')
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import csv
from IPython.display import Image, display
from dataloading import *
from utils.class_mapping import stem_dir 
from model_init import *
from analysis.eval_model import *
from train import *
from lcapt.lca import LCAConv2D
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
# Data
data_dir = '/net/smtcac0060/fs2/scr/scr2/datasets/cifar/' # '/net/smtcac2623/fs1/scr/scr1/mushtaq/cifar/' 

# Model
model_name = "LCA CIFAR subset"                # model architecture which shall be used
num_classes = 10                                    # number of classes to be differentiated
# Training
batch_size = 128                                     # size of training and validation batches
num_epochs_LCA = 25                                  # num of epochs for training
lr = 0.001                                          # learning rate ResNet
eta = 0.01                                          # learning rate lca layer 
weight_decay = 0.0

FEATURES = 128                                       # number of dictionary features to learn M
KERNEL_SIZE = 7                                     # height and width of each kernel
LAMBDA = 5                                        # LCA threshold
LCA_ITERS = 600
print_freq = 1
STRIDE = 2                                          # convolutional stride 
TAU = 100                                           # LCA time constant T


# paths
model_dir = stem_dir + "/models/_params/subset-CIFAR/" + "lca_dict" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}_iters_{LCA_ITERS}_tau_{TAU}" + ".pt"
result_dir = stem_dir + "/models/_params/subset-CIFAR/" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"
metrics_path = result_dir + "/metrics.csv"
metrics_plot_path = stem_dir + "/figures/subset-CIFAR" + "/LCA_train_plots"

fig_path_acc = stem_dir + "/figures/subset-CIFAR/Acc_over_epochs_lca_frontend" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
fig_path_loss = stem_dir + "/figures/subset-CIFAR/Loss_over_epochs" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
fig_path_cm = stem_dir + "/figures/subset-CIFAR/conf_matrix_lca_frontend" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
fig_path_lca = stem_dir + "/figures/subset-CIFAR/lca_training" + "_lr_" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
fig_path_lca_recon = stem_dir + "/figures/subset-CIFAR/lca_recon" + "_lr_" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
fig_path_lca_features = stem_dir + "/figures/subset-CIFAR/lca_features" + "_lr_" + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}" + ".png"
os.makedirs(os.path.dirname(fig_path_lca_recon), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_lca_features), exist_ok=True)

""" Define model """
class LCA_CIFAR(nn.Module):
    def __init__(self):
        super().__init__()
        self.LCA = LCAConv2D(
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
            track_metrics=True,
            return_vars=['inputs', 'acts', 'recons', 'recon_errors'],
            transfer_func="hard_threshold",
        )


    def forward(self, x):
        #  Forward pass through the LCA layer
        inputs, code, x1, recon_error = self.LCA(x)
   
        return inputs, code, x1, recon_error, x


if __name__ == '__main__':
    
    """ Check for availability of GPU """
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    
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
    
    # Create training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
    
    """ Load model """
    model = LCA_CIFAR()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay) 
 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    """ Train LCANet """
    l1, l2, energy = [], [], []
    
    # train LCA layer
    for epoch in range(num_epochs_LCA):
        if (epoch + 1) % 4 == 0:
            model.LCA.lambda_ += 0.1
            
        for batch_num, (images, _) in enumerate(train_loader):
            images = images.to(device)
            inputs, code, recon, recon_error, x = model(images)
            model.LCA.update_weights(code, recon_error)
            
            # generate analytics data of training progress
            if batch_num % print_freq == 0:
                l1_sparsity = compute_l1_sparsity(code, model.LCA.lambda_).item()
                l2_recon_error = compute_l2_error(inputs, recon).item()
                total_energy = l2_recon_error + l1_sparsity
                print(f'L2 Recon Error: {round(l2_recon_error, 2)}; ',
                      f'L1 Sparsity: {round(l1_sparsity, 2)}; ',
                      f'Total Energy: {round(total_energy, 2)}')
                l1.append(l1_sparsity)
                l2.append(l2_recon_error)
                energy.append(total_energy)                                                                                                            

                
    """ Show lca properties """
    plot_lca_recon(model, test_loader, device, fig_path_lca_recon)
    plot_lca_features(model, fig_path_lca_features)
        
    
    """ Save model """
    torch.save(model.LCA.state_dict(), model_dir)
    
    train_acc_history, val_acc_history, train_loss_history, val_loss_history = [], [], [],[]
    """ Visualization Training """
    train_params=[lr, weight_decay, batch_size, 0.0]
    lca_params = [FEATURES, LAMBDA, KERNEL_SIZE, lr, LCA_ITERS, STRIDE, TAU]
    plot_model_perf(model_name, train_params, lca_params, train_acc_history, val_acc_history, train_loss_history, val_loss_history, num_epochs_LCA,
                    fig_path_acc, fig_path_loss, "lca", l1, l2, energy, fig_path_lca)
    

    """ Confusion matrix for validation data """
    # plt_confusion_matrix(val_dataloader, model, "torch", device, batch_size, fig_path_cm)    
# %%

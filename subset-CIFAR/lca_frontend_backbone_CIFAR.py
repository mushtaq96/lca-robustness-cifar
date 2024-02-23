# -*- coding: utf-8 -*-
# %%
from __future__ import print_function
from __future__ import division
import random
import sys

sys.path.append("/home/bokhars/thesis/robustness/")
import torch
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

from lcapt.analysis import make_feature_grid
from lcapt.lca import LCAConv1D, LCAConv2D
from lcapt.metric import compute_l1_sparsity, compute_l2_error
from lcapt.preproc import make_unit_var, make_zero_mean

# setting the seed value for reproducibiity
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = (
    True  # deterministic behavior for cuDNN to make GPU computations reproducible
)
torch.backends.cudnn.benchmark = False  # disable cuDNN benchmarking, which can introduce variability in GPU computations

""" Inputs """
# Data
data_dir = "/net/smtcac0060/fs2/scr/scr2/datasets/cifar/"  # '/net/smtcac2623/fs1/scr/scr1/mushtaq/tiny-imagenet-200'

# Model
model_name = (
    "LCA Frontend + Scratchmodel CIFAR"  # model architecture which shall be used
)
num_classes = 10  # number of classes to be differentiated
# Training
batch_size = 128  # size of training and validation batches
num_epochs = 20
lr = 0.1  # learning rate ResNet
eta = 0.01  # learning rate lca layer
weight_decay = 0.0002

FEATURES = 64  # number of dictionary features to learn
KERNEL_SIZE = 7  # height and width of each kernel
TRAIN_LAMBDA = 0.8  # LCA threshold
LAMBDA = 0.5  # this is the most clear kernel
LCA_ITERS = 100
print_freq = 1
STRIDE = 2  # convolutional stride
TAU = 300  # LCA time constant

# paths
model_dir = (
    stem_dir
    + "/models/_params/subset-CIFAR/lca_frontend_backbone/"
    + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"
    + ".pt"
)
result_dir = (
    stem_dir
    + "/models/_params/subset-CIFAR/lca_frontend_backbone/"
    + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"
)

fig_path = stem_dir + "/figures/subset-CIFAR/lca_frontend_backbone/"
fig_path_inference = (
    stem_dir
    + "/figures/subset-CIFAR/lca_frontend_backbone/Inference_"
    + f"_lr_{lr}_bs_{batch_size}"
    + ".png"
)
fig_path_acc = (
    stem_dir
    + "/figures/subset-CIFAR/lca_frontend_backbone/Acc_over_epochs_lca_frontend"
    + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"
    + ".png"
)
fig_path_loss = (
    stem_dir
    + "/figures/subset-CIFAR/lca_frontend_backbone/Loss_over_epochs"
    + f"_lr_{lr}_eta_{eta}_lambda_{LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}"
    + ".png"
)


os.makedirs(os.path.dirname(model_dir), exist_ok=True)
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_acc), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_loss), exist_ok=True)
os.makedirs(os.path.dirname(fig_path_inference), exist_ok=True)

LCA_model_path = (
    stem_dir
    + "/models/_params/subset-CIFAR/"
    + "lca_dict"
    + f"_lr_0.001_eta_{eta}_lambda_{TRAIN_LAMBDA}_ftr_{FEATURES}_size_{KERNEL_SIZE}_stride_{STRIDE}_iters_{LCA_ITERS}_tau_{TAU}"
    + ".pt"
)
scratchmodel_path = (
    stem_dir
    + "/models/_params/subset-CIFAR/backbone_CIFAR/"
    + f"backbone_CIFAR_lr_{lr}_bs_{batch_size}"
    + ".pt"
)

""" Define model """


class LCA_Frontend_Backbone_CIFAR(nn.Module):
    def __init__(self, lca_model, resnet18, num_classes):
        super().__init__()
        self.LCA = lca_model
        self.decoder = resnet18
        self.decoder.conv1 = nn.Conv2d(
            FEATURES, 64, 7, 2, 3
        )  # (input channels, output channels, kernel size, padding) Why FEATURES should it not be 3?
        self.decoder.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        #  Forward pass through the LCA layer
        inputs, code, x1, recon_error = self.LCA(x)
        # Forward pass through the modified ResNet18 decoder
        x = self.decoder(code)
        return x


# %%
if __name__ == "__main__":

    """Check for availability of GPU"""
    print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))

    categories = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    class_dict = {}
    for i in range(len(categories)):
        class_dict[i] = categories[i]

    """ Load Data """
    path = "/home/bokhars/thesis/robustness/architectures/subset-CIFAR"
    train_dataset = torch.load(path + "/cifar10-train.pth")
    test_dataset = torch.load(path + "/cifar10-test.pth")

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
    print("Training subset class counts:", train_class_counts)
    print("Testing subset class counts:", test_class_counts)

    # Create training and validation dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

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
        return_vars=["inputs", "acts", "recons", "recon_errors"],
        transfer_func="hard_threshold",
    )

    """ Load Resnet18 model """
    resnet18 = models.resnet18(weights=None)
    print(resnet18)

    resnet18.load_state_dict(torch.load(scratchmodel_path))

    lca_pretrained.load_state_dict(torch.load(LCA_model_path))

    weight_grid = make_feature_grid(lca_pretrained.get_weights())
    plt.imshow(weight_grid.float().cpu().numpy())

    model = LCA_Frontend_Backbone_CIFAR(
        lca_model=lca_pretrained, resnet18=resnet18, num_classes=num_classes
    )
    model = model.to(device)

    # filer used because only the unfrozen parameters are to be updated during training
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )

    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    """ Train Model """
    train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist = train_model(
        model,
        device,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        scheduler=scheduler,
    )

    # Plot training and model performance
    train_params = [lr, weight_decay, batch_size, num_epochs]
    lca_params = [None]
    plot_model_perf(
        model_name,
        train_params,
        lca_params,
        train_acc_hist,
        val_acc_hist,
        train_loss_hist,
        val_loss_hist,
        num_epochs=num_epochs,
        fig_path_acc=fig_path_acc,
        fig_path_loss=fig_path_loss,
    )

    """ Inference of model"""
    plot_inference_results_custom_model(
        test_loader, model, class_dict, device=device, fig_path=fig_path_inference
    )

    # """ Save model """
    # torch.save(model.state_dict(), model_dir)

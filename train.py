# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:07:52 2023

@author: jakob
"""
import torch
import time
import copy
import math
import matplotlib.pyplot as plt
import os
from analysis.eval_model import *
from tqdm import tqdm
from random import shuffle
# from art.attacks.evasion import ProjectedGradientDescentPyTorch
# from art.data_generators import PyTorchDataGenerator
# from art.estimators.classification import PyTorchClassifier
# from art.defences.trainer import AdversarialTrainerMadryPGD
from lcapt.metric import compute_l1_sparsity, compute_l2_error

"""
----------------------------------------------------------------------------------------------------------------
set_parameter_requires_grad
determining whether gradients shall be calculated for model parameters --> used for freezing layers during fine tuning
---------------------------
inputs:
    model               --> model, which shall be trained/evaluated
    feature_extracting  --> flag, whether gradient calculation shall be for this tensor --> True for finetuning
---------------------------
output:
                        --> returns model with frozen gradients
----------------------------------------------------------------------------------------------------------------
"""
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

"""
----------------------------------------------------------------------------------------------------------------
vis_lr_schedule
visualize learning rate schedule over epochs
---------------------------
inputs:
    optimizer           --> optimizer used for training
    schedule            --> name of scheduling method
    num_epochs          --> number of epochs
    step                --> step rate for step learning
    gamma               --> decay factor
---------------------------
output:
                
                        --> line graph of learning rate over epochs
----------------------------------------------------------------------------------------------------------------    
"""
def vis_lr_schedule(optimizer, scheduler, num_epochs, step=None, gamma=None):
    # initialize learning rate and lists for visualization
    init_lr = optimizer.param_groups[0]['lr']
    lr = []
    epochs = []
            
    # ExponentialLR: calculate learning rate for each epoch based on scheduler form
    for i in range(num_epochs):
        epochs.append(i)
        # Step decay
        if scheduler == "StepLR":
            if i == 0:
                lr.append(init_lr)
            elif i%step == 0: 
                init_lr = lr[-1]*gamma
                lr.append(init_lr)
            else:
                lr.append(init_lr)
        # Exponential learning rate decay    
        elif scheduler == "ExpLR":
            if i == 0:
                lr.append(init_lr)
            else:
                lr.append(init_lr*math.exp(-gamma*i))
                
    # visualize learning rate
    plt.plot(epochs, lr)
    plt.title('learning rate over epochs')
    plt.xlabel('epochs')
    plt.ylabel('learning rate')
    plt.show()

def adjust_learning_rate(optimizer, epoch):
    lr = 0.1
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

"""
----------------------------------------------------------------------------------------------------------------
train_model
takes a model and trains it on training data, while evaluating it on validation data
---------------------------
inputs:
    model               --> model of a neural network taken for training and validation
    train_dataloader    --> dataloader providing batches of training data
    val_dataloader      --> dataloader providing batches of validation data
    criterion           --> Loss function
    optimizier          --> Opimizer taken for gradient descent
    num_epochs          --> number of epochs taken for training
    scheduler           --> learning rate scheduler in case of usage 
---------------------------
output:
    model               --> outputs trained model, taking the best version of all epochs
    train_acc_history   --> list of training accuracy per epoch (used for plotting)
    val_acc_history     --> list of validation accuracy per epoch (used for plotting)
    train_loss_history  --> list of training loss per epoch (used for plotting)
    val_loss_history    --> list of validation loss per epoch (used for plotting)
----------------------------------------------------------------------------------------------------------------
"""
def train_model(model, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, scheduler=None):
    since = time.time()

    # initialize variables for training and validation history
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, epoch) # for cifar only
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0

        ##train 
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False, desc="Processing Train data"):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))

        running_loss = 0.0
        running_corrects = 0

        ##VAL
        with torch.no_grad():
            for inputs, labels in tqdm(val_dataloader, leave=False, desc="Processing Val data"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # run scheduler in case of ReduceLROnPlateau
                if scheduler:
                    scheduler.step()

            epoch_loss = running_loss / len(val_dataloader.dataset)
            epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))

            val_acc_history.append(epoch_acc)
            val_loss_history.append(epoch_loss)

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))


    return train_acc_history, val_acc_history, train_loss_history, val_loss_history


"""
----------------------------------------------------------------------------------------------------------------
train_pgd_model
takes a model and trains it on training data, while evaluating it on validation data
---------------------------
inputs:
    model               --> model of a neural network taken for training and validation
    input_shape         --> tuple describing shape of input images 
    num_classes         --> integer for number of classes of classifier
    batch_size          --> batch_size
    device              --> device on which training takes place
    train_dataloader    --> dataloader providing batches of training data
    val_dataloader      --> dataloader providing batches of validation data
    criterion           --> Loss function
    optimizier          --> Opimizer taken for gradient descent
    num_epochs          --> number of epochs taken for training
    eps                 --> hyperparameter epsilon as maximum perturbation of adv. examples
    eps_step            --> step size for pgd perturbation
    max_iter            --> maximum number of steps for pgd attack
    ratio               --> ratio of images per batch to add as adversarial examples to training data
    norm                --> distance norm used for pgd attack
---------------------------
output:
    model               --> outputs trained model, taking the best version of all epochs
    train_acc_history   --> list of training accuracy per epoch (used for plotting)
    val_acc_history     --> list of validation accuracy per epoch (used for plotting)
    train_loss_history  --> list of training loss per epoch (used for plotting)
    val_loss_history    --> list of validation loss per epoch (used for plotting)
----------------------------------------------------------------------------------------------------------------
"""

def train_pgd_model(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, eps=0.01, eps_step=0.01/35, max_iter=50, ratio=1.0, norm="inf"):
    since = time.time()
        
    # initialize variables for training and validation history
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    adv_batch_size = round(batch_size*ratio)
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
 
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0

        model.train()  # Set model to training mode
        
        # create adversarial model
        classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        input_shape=input_shape,
        optimizer=optimizer,
        nb_classes=num_classes,
        device_type="gpu",
        )
            
        # setup attack
        attack = ProjectedGradientDescentPyTorch(
            estimator=classifier,
            norm=norm,
            eps=eps,  
            eps_step=eps_step,
            max_iter=max_iter)
   
        # introduce batch count to control adversarial example creation
        batch_cnt = 0
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            
            # Generate batch of adversarial examples
            if epoch > 0 or batch_cnt > 0:
                # create adversarial examples
                adv_inputs = attack.generate(x=inputs[0:adv_batch_size].numpy())
                adv_inputs = torch.from_numpy(adv_inputs)
                
                # concat clean and adversarial data
                inputs = torch.cat((inputs, adv_inputs), 0)
                labels = torch.cat((labels, labels[0:adv_batch_size]), 0)
                
                # create list of random values to shuffle labels and images
                shuffle_list = [i for i in range(0, batch_size+adv_batch_size)]
                shuffle(shuffle_list)
                shuffle_list = torch.Tensor(shuffle_list).type(torch.int64)

                # 
                new_labels = torch.zeros(batch_size+adv_batch_size, dtype=torch.int64)       
                new_inputs = torch.zeros((batch_size+adv_batch_size, 3, input_shape[0], input_shape[1]), dtype=torch.float64)
                for i in range(labels.size(dim=0)):
                    new_labels[shuffle_list[i]] = labels[i]
                    new_inputs[shuffle_list[i]] = inputs[i]
                    
            batch_cnt +=1
                                
            # Take training batch    
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
           
        # calculate loss and accuracy over epoch and add to history --> add length dataset with ratio to accompensate for added aversarial examples
        epoch_loss = running_loss / (len(train_dataloader.dataset)+(round(len(train_dataloader.dataset)*ratio)))
        epoch_acc = running_corrects.double() / (len(train_dataloader.dataset)+(round(len(train_dataloader.dataset)*ratio)))
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
            
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
           
        model.eval()   # Set model to evaluate mode
           
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
           
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
                   
            with torch.set_grad_enabled(False):
                   
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                   
                _, preds = torch.max(outputs, 1)
                   
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
        

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history


"""
----------------------------------------------------------------------------------------------------------------
train_lca_model
takes a model and trains it on training data, while evaluating it on validation data
---------------------------
inputs:
    model               --> model of a neural network taken for training and validation
    input_shape         --> tuple describing shape of input images 
    num_classes         --> integer for number of classes of classifier
    batch_size          --> batch_size
    device              --> device on which training takes place
    train_dataloader    --> dataloader providing batches of training data
    val_dataloader      --> dataloader providing batches of validation data
    criterion           --> Loss function
    optimizier          --> Opimizer taken for gradient descent
    num_epochs_LCA      --> number of epochs taken for LCA training
    num_epochs_ResNet   --> number of epochs taken for ResNet training
    print_freq          --> count of batch after which training progress shall be output
    metrics_path        --> optional: path where LCA metrics are stored as csv
    metrics_plot_path   --> optional: path where LCA training metrics are plotted
    scheduler           --> optional: learning rate scheduler 
---------------------------
output:
    model               --> outputs trained model, taking the best version of all epochs
    l1                  --> list containing l1 sparsity values of training (used for plotting)
    l2                  --> list containing l2 reconstruction error of training (used for plotting)
    energy              --> list containing values of energy function of training (used for plotting)
    train_acc_history   --> list of training accuracy per epoch (used for plotting) 
    val_acc_history     --> list of validation accuracy per epoch (used for plotting)
    train_loss_history  --> list of training loss per epoch (used for plotting)
    val_loss_history    --> list of validation loss per epoch (used for plotting)
----------------------------------------------------------------------------------------------------------------
"""

def train_lca_model(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs_lca, num_epochs_ResNet, print_freq=1, 
                    metrics_path=None, metrics_plot_path=None, scheduler=None):
    """ Train LCANet """
    l1, l2, energy = [], [], []
    
    # Freeze ResNet and train LCA Layer
    model.decoder.requires_grad_(False)
    
    # cleanup existing LCA metrics
    if model.LCA.track_metrics == True:
        if os.path.exists(metrics_path) == True:
            os.remove(metrics_path)
    
    # train LCA layer
    for epoch in range(num_epochs_lca):
        for batch_num, (images, _) in enumerate(train_dataloader):
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
                
                
    """ Evaluate LCA Training """ 
    if model.LCA.track_metrics == True and metrics_path != None and metrics_plot_path != None:
        # figures per batch
        metrics_path = metrics_path
        metrics_plot_path = metrics_plot_path
        if os.path.exists(metrics_plot_path) == False:
            os.mkdir(metrics_plot_path)
            print("creating directory")
        store_lca_metrics(metrics_path, str(metrics_plot_path + "/"))
        
        # remove data after storing plots
        os.remove(metrics_path)
        
        print("stored metrics")

    # disable metrics tracking
    model.LCA.track_metrics = False

    """ train Resnet18 """
    since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs_ResNet):
        print('Epoch {}/{}'.format(epoch, num_epochs_ResNet - 1))
        print('-' * 10)
    
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0
    
        model.train()  # Set model to training mode
    
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # disable for LCA Layer and enable gradients for ResNet
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.decoder.parameters():
                param.requires_grad = True
                
            # Get model outputs and calculate loss
            outputs = model(inputs)[4]
            loss = criterion(outputs, labels)
    
            _, preds = torch.max(outputs, 1)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        model.eval()   # Set model to evaluate mode
        
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # turn gradient tracking off
            for param in model.parameters():
                param.requires_grad = False
                
            with torch.set_grad_enabled(False):
                
                outputs = model(inputs)[4]
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        # run scheduler in case of ReduceLROnPlateau
        if scheduler:
            scheduler.step()
    
        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))
    
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
    
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, l1, l2, energy, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def train_LCA_firstL_model(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs_lca, num_epochs_ResNet, print_freq=1, 
                    metrics_path=None, metrics_plot_path=None, scheduler=None):
    """ Train LCANet """
    l1, l2, energy = [], [], []
    
    # Freeze ResNet and train LCA Layer which replaced conv1 of resnet
    # Freeze the weights of the pre-trained layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the weights of the LCAConv2D layer
    for param in model.conv1.parameters():
        param.requires_grad = True

    
    # train LCA layer
    for epoch in range(num_epochs_lca):
        for batch_num, (images, _) in enumerate(train_dataloader):
            images = images.to(device)
            inputs, code, recon, recon_error, x = model(images) # the output of the 
            model.conv1[0].update_weights(code, recon_error)
            
            # generate analytics data of training progress
            if batch_num % print_freq == 0:
                l1_sparsity = compute_l1_sparsity(code, model.conv1[0].lambda_).item()
                l2_recon_error = compute_l2_error(inputs, recon).item()
                total_energy = l2_recon_error + l1_sparsity
                print(f'L2 Recon Error: {round(l2_recon_error, 2)}; ',
                      f'L1 Sparsity: {round(l1_sparsity, 2)}; ',
                      f'Total Energy: {round(total_energy, 2)}')
                l1.append(l1_sparsity)
                l2.append(l2_recon_error)
                energy.append(total_energy)

    """ train Resnet18 """
    since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs_ResNet):
        print('Epoch {}/{}'.format(epoch, num_epochs_ResNet - 1))
        print('-' * 10)
    
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0
    
        model.train()  # Set model to training mode
    
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # disable for LCA Layer and enable gradients for ResNet
            for param in model.parameters():
                param.requires_grad = False
            
            
            for param in model.resnet_layers.parameters():
                param.requires_grad = True
                
                
            # Get model outputs and calculate loss
            outputs = model(inputs)[4]
            loss = criterion(outputs, labels)
    
            _, preds = torch.max(outputs, 1)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        model.eval()   # Set model to evaluate mode
        
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # turn gradient tracking off
            for param in model.parameters():
                param.requires_grad = False
                
            with torch.set_grad_enabled(False):
                
                outputs = model(inputs)[4]
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        # run scheduler in case of ReduceLROnPlateau
        if scheduler:
            scheduler.step()
    
        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))
    
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
    
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, l1, l2, energy, train_acc_history, val_acc_history, train_loss_history, val_loss_history


"""
----------------------------------------------------------------------------------------------------------------
train_pretrained_lca_model
takes a model and trains it on training data, while evaluating it on validation data
---------------------------
inputs:
    model               --> model of a neural network taken for training and validation
    input_shape         --> tuple describing shape of input images 
    num_classes         --> integer for number of classes of classifier
    batch_size          --> batch_size
    device              --> device on which training takes place
    train_dataloader    --> dataloader providing batches of training data
    val_dataloader      --> dataloader providing batches of validation data
    criterion           --> Loss function
    optimizier          --> Opimizer taken for gradient descent
    num_epochs_ResNet   --> number of epochs taken for ResNet training
    LAMBDA              --> lambda used for forward pass
    scheduler           --> optional: learning rate scheduler 
---------------------------
output:
    model               --> outputs trained model, taking the best version of all epochs
    train_acc_history   --> list of training accuracy per epoch (used for plotting) 
    val_acc_history     --> list of validation accuracy per epoch (used for plotting)
    train_loss_history  --> list of training loss per epoch (used for plotting)
    val_loss_history    --> list of validation loss per epoch (used for plotting)
----------------------------------------------------------------------------------------------------------------
"""

def train_pretrained_lca_model(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs_ResNet, LAMBDA, scheduler=None):

    # set lambda for forward pass
    model.LCA.lambda_ = LAMBDA    

    # disable metrics tracking
    model.LCA.track_metrics = False

    """ train Resnet """
    since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs_ResNet):
        print('Epoch {}/{}'.format(epoch, num_epochs_ResNet - 1))
        print('-' * 10)
    
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0
    
        model.train()  # Set model to training mode
    
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # disable for LCA Layer and enable gradients for ResNet
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.decoder.parameters():
                param.requires_grad = True
                
            # Get model outputs and calculate loss
            outputs = model(inputs)[4]
            # print("Output shape:", outputs.shape)
            # print("Label shape:", labels.shape)
            loss = criterion(outputs, labels)
    
            _, preds = torch.max(outputs, 1)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        model.eval()   # Set model to evaluate mode
        
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # turn gradient tracking off
            for param in model.parameters():
                param.requires_grad = False
                
            with torch.set_grad_enabled(False):
                
                outputs = model(inputs)[4]
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        # run scheduler in case of ReduceLROnPlateau
        if scheduler:
            scheduler.step()
    
        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))
    
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
    
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history


def train_pretrained_lcafirstL_model(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs_ResNet, LAMBDA, scheduler=None):

    # set lambda for forward pass
    model.conv1.lambda_ = LAMBDA    

    # disable metrics tracking
    model.conv1.track_metrics = False

    """ train Resnet """
    since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs_ResNet):
        print('Epoch {}/{}'.format(epoch, num_epochs_ResNet - 1))
        print('-' * 10)
    
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0
    
        model.train()  # Set model to training mode
    
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # disable for LCA Layer and enable gradients for ResNet
            for param in model.parameters():
                param.requires_grad = True
            
            for param in model.conv1.parameters():
                param.requires_grad = False
                
            # Get model outputs and calculate loss
            outputs = model(inputs)[4]
            loss = criterion(outputs, labels)
    
            _, preds = torch.max(outputs, 1)
    
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        model.eval()   # Set model to evaluate mode
        
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # turn gradient tracking off
            for param in model.parameters():
                param.requires_grad = False
                
            with torch.set_grad_enabled(False):
                
                outputs = model(inputs)[4]
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        # run scheduler in case of ReduceLROnPlateau
        if scheduler:
            scheduler.step()
    
        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))
    
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
    
    print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history


"""
----------------------------------------------------------------------------------------------------------------
train_lca_fully_connected
takes a model and trains it on training data, while evaluating it on validation data
---------------------------
inputs:
    model               --> model of a neural network taken for training and validation
    input_shape         --> tuple describing shape of input images 
    num_classes         --> integer for number of classes of classifier
    batch_size          --> batch_size
    device              --> device on which training takes place
    train_dataloader    --> dataloader providing batches of training data
    val_dataloader      --> dataloader providing batches of validation data
    criterion           --> Loss function
    optimizier          --> Opimizer taken for gradient descent
    num_epochs_LCA      --> number of epochs taken for LCA training
    num_epochs_ResNet   --> number of epochs taken for ResNet training
    print_freq          --> count of batch after which training progress shall be output
    scheduler           --> optional: learning rate scheduler 
---------------------------
output:
    model               --> outputs trained model, taking the best version of all epochs
    l1                  --> list containing l1 sparsity values of training (used for plotting)
    l2                  --> list containing l2 reconstruction error of training (used for plotting)
    energy              --> list containing values of energy function of training (used for plotting)
    train_acc_history   --> list of training accuracy per epoch (used for plotting) 
    val_acc_history     --> list of validation accuracy per epoch (used for plotting)
    train_loss_history  --> list of training loss per epoch (used for plotting)
    val_loss_history    --> list of validation loss per epoch (used for plotting)
----------------------------------------------------------------------------------------------------------------
"""

def train_lca_fully_connected(model, input_shape, num_classes, batch_size, device, train_dataloader, val_dataloader, criterion, optimizer, num_epochs_LCA, 
                              num_epochs_ResNet, print_freq=1, metrics_path=None, metrics_plot_path=None, scheduler=None, lambda_schedule=None, target_lambda=None):
    
    """ Train LCA Layer """
    l1, l2, energy = [], [], []
    
    model.backbone.requires_grad_(False)
    
    # cleanup existing LCA metrics
    if model.LCA.track_metrics == True:
        if os.path.exists(metrics_path) == True:
            os.remove(metrics_path)
    
    for epoch in range(num_epochs_LCA):
        for batch_num, (images, _) in enumerate(train_dataloader):
            images = images.to(device)
            inputs, code, recon, recon_error, x = model(images)
            model.LCA.update_weights(code, recon_error)
            
            if lambda_schedule:
                if batch_num % lambda_schedule == 0:
                    model.LCA.lambda_+=0.1
            
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
                
    if target_lambda:
        model.LCA.lambda_ = target_lambda
    
    
    """ Evaluate LCA Training """ 
    if model.LCA.track_metrics == True and metrics_path != None and metrics_plot_path != None:
        # figures per batch
        metrics_path = metrics_path
        metrics_plot_path = metrics_plot_path
        if os.path.exists(metrics_plot_path) == False:
            os.mkdir(metrics_plot_path)
            print("creating directory")
        store_lca_metrics(metrics_path, str(metrics_plot_path + "\\"))
        
        # remove metrics after storing plots
        os.remove(metrics_path)
        
        print("stored metrics")
    
    # disable tracking metrics after storing training plots
    model.LCA.track_metrics = False
    
    """ Train fully connected layer """
    # train Resnet 
    since = time.time()
    
    train_acc_history = []
    val_acc_history = []
    train_loss_history = []
    val_loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
        
    for epoch in range(num_epochs_ResNet):
        print('Epoch {}/{}'.format(epoch, num_epochs_ResNet - 1))
        print('-' * 10)
    
        # Setup loss for training phase
        running_loss = 0.0
        running_corrects = 0
    
        model.train()  # Set model to training mode
    
        # Iterate over data.
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward
            # disable gradients for model
            for param in model.parameters():
                param.requires_grad = False
                
            # enable gradients for fully connected layer
            for param in model.fc.parameters():
                param.requires_grad = True
            
            # Get model outputs and calculate loss
            outputs = model(inputs)[4]
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
            
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # calculate loss and accuracy over epoch and add to history
        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(train_dataloader.dataset)
        train_acc_history.append(epoch_acc)
        train_loss_history.append(epoch_loss)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("train", epoch_loss, epoch_acc))
        
        
        # clear loss for validation phase
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_dataloader, leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # turn gradient tracking off
            for param in model.parameters():
                param.requires_grad = False
                
            with torch.set_grad_enabled(False):
                
                outputs = model(inputs)[4]
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                
            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            

    
        epoch_loss = running_loss / len(val_dataloader.dataset)
        epoch_acc = running_corrects.double() / len(val_dataloader.dataset)
    
        print('{} Loss: {:.4f} Acc: {:.4f}'.format("val", epoch_loss, epoch_acc))
    
        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        val_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_loss)
    
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
    
    
    return model, l1, l2, energy, train_acc_history, val_acc_history, train_loss_history, val_loss_history

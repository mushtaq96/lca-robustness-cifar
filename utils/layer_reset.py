import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models

def reset_weights(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        if layer.bias is not None:
            init.constant_(layer.bias, 0)

def reset_weights_xavier(layer):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        # Compute the mean and standard deviation of the weight tensor before reset
        weight_mean_before = layer.weight.mean().item()
        weight_std_before = layer.weight.std().item()

        init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            init.constant_(layer.bias, 0) # remove this because none of the conv layers have bias
        
        # Compute the mean and standard deviation of the weight tensor after reset
        weight_mean_after = layer.weight.mean().item()
        weight_std_after = layer.weight.std().item()

        # Print the mean and standard deviation before and after reset
        print("Weight mean before reset:", weight_mean_before)
        print("Weight mean after reset:", weight_mean_after)
        print("Weight std before reset:", weight_std_before)
        print("Weight std after reset:", weight_std_after)
            
def sanity_check(model):
    print("Sanity Checking\n")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print('\033[91m' + f'{name} is frozen' + '\033[0m')
        else:
            print('\033[32m' + f'{name} is not frozen' + '\033[0m')

def decide_reset_1(model, layers_to_unfreeze):
    # freeze all layers of the model
    for param in model.parameters():
        # indicates whether parameter's gradients should be calculated
        param.requires_grad = False

    # unfreeze specific layers
    for name, child in model.named_children():
        if name == 'conv1':
            for param in child.parameters():
                print('layer name:', name)
                print('child:', child)

                # parameter's gradients are calculated
                param.requires_grad = True
                reset_weights_xavier(child)  # reset 

        if name == 'layer1':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'layer2':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'layer3':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
                    
        if name == 'layer4':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'fc' and 'fc.weight' in layers_to_unfreeze:
            for name, param in child.named_parameters():
                if name == 'weight':
                    param.requires_grad = True 
            reset_weights_xavier(child)  # reset 
                
    # print("inside decide_reset function, model : ", model)
    sanity_check(model)
    return model

def decide_reset(model, layers_to_unfreeze):
    # Freeze all layers of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers
    for name, child in model.named_children():
        for s in layers_to_unfreeze:
            if name in s:
                for param in child.parameters():
                    param.requires_grad = True
            
                reset_weights_xavier(child)

    # Print the layers that are unfrozen
    print("Unfrozen layers:", layers_to_unfreeze)
    sanity_check(model)
    return model

def decide_unfreeze(model, layers_to_unfreeze):
    # Freeze all layers of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers
    for name, child in model.named_children():
        for s in layers_to_unfreeze:
            if name in s:
                for param in child.parameters():
                    param.requires_grad = True

    # Print the layers that are unfrozen
    print("Unfrozen layers:", layers_to_unfreeze)
    sanity_check(model)
    return model

def decide_unfreeze1(model, layers_to_unfreeze):
    # freeze all layers of the model
    for param in model.parameters():
        # indicates whether parameter's gradients should be calculated
        param.requires_grad = False

    # unfreeze specific layers
    for name, child in model.named_children():
        if name == 'conv1':
            for param in child.parameters():
                print('layer name:', name)
                print('child:', child)

                # parameter's gradients are calculated
                param.requires_grad = True

        if name == 'layer1':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                            
        if name == 'layer2':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                            
        if name == 'layer3':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
        if name == 'layer4':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.BasicBlock):
                    for param in block.conv2.parameters():
                        param.requires_grad = True

        if name == 'fc' and 'fc.weight' in layers_to_unfreeze:
            for name, param in child.named_parameters():
                if name == 'weight':
                    param.requires_grad = True                 
        
    # print("inside decide_reset function, model : ", model)
    sanity_check(model)
    return model



def decide_unfreeze_r50(model, layers_to_unfreeze):
    # freeze all layers of the model
    for param in model.parameters():
        # indicates whether parameter's gradients should be calculated
        param.requires_grad = False

    # unfreeze specific layers
    for name, child in model.named_children():
        if name == 'conv1':
            for param in child.parameters():
                print('layer name:', name)
                print('child:', child)

                # parameter's gradients are calculated
                param.requires_grad = True

        if name == 'layer1':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                            
        if name == 'layer2':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                            
        if name == 'layer3':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
        if name == 'layer4':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True

        if name == 'fc' and 'fc.weight' in layers_to_unfreeze:
            for name, param in child.named_parameters():
                if name == 'weight':
                    param.requires_grad = True                 
        
    # print("inside decide_reset function, model : ", model)
    sanity_check(model)
    return model


def decide_reset_r50(model, layers_to_unfreeze):
    # freeze all layers of the model
    for param in model.parameters():
        # indicates whether parameter's gradients should be calculated
        param.requires_grad = False

    # unfreeze specific layers
    for name, child in model.named_children():
        if name == 'conv1':
            for param in child.parameters():
                print('layer name:', name)
                print('child:', child)

                # parameter's gradients are calculated
                param.requires_grad = True
                reset_weights_xavier(child)  # reset 

        if name == 'layer1':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'layer2':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'layer3':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
                    
        if name == 'layer4':
            for block_name, block in child.named_children():
                if f"{name}.{block_name}.conv1" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):

                    for param in block.conv1.parameters():
                        param.requires_grad = True

                    reset_weights_xavier(block.conv1)

                if  f"{name}.{block_name}.conv2" in layers_to_unfreeze and isinstance(block, models.resnet.Bottleneck):
                    for param in block.conv2.parameters():
                        param.requires_grad = True
                    
                    reset_weights_xavier(block.conv2)
        
        if name == 'fc' and 'fc.weight' in layers_to_unfreeze:
            for name, param in child.named_parameters():
                if name == 'weight':
                    param.requires_grad = True 
            reset_weights_xavier(child)  # reset 
                
    # print("inside decide_reset function, model : ", model)
    sanity_check(model)
    return model

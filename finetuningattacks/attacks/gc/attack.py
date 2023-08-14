# code for performing gradient canceling attack on logistic regression

import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from numpy import linalg as LA
import numpy as np
import math
import torchvision.models as models
from torchvision.models import resnet


def attack_gc(
    poisoned_model, #
    poisoned_model_steps,
    poisoned_model_optimizer, #
    train_loss,
    test_loss,
    eval_metric,
    train_loader, #
    test_loader, #
    train_ratio,
    epsilon, #
    epochs, #
    print_epochs, 
    save_epochs,
    save_folder,
    device, #
):

    torch.manual_seed(0)

    # hyperparameters
    lr = 1e5
    resume = True


    # load corrupted model
    model = poisoned_model.to(device)
    poisoned_model.load_state_dict(torch.load(os.path.join(save_folder, "cifar10_resnet.pt")))
    optimizer = poisoned_model_optimizer(poisoned_model.head.parameters())


    def autograd(outputs, inputs, create_graph=False):
        """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
        #inputs = tuple(inputs)
        grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
        return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]

    # if resume==False:
    #     print("=====> Start calculating the clean gradients")
    #     total_grad_clean = torch.zeros(10,512)
    #     for data, target in pre_loader:
    #         data, target = data.to(device), target.to(device).long()
    #         criterion = nn.CrossEntropyLoss(reduction='sum')

    #         # calculate gradient of w on clean sample
    #         output_c = model(data)
    #         loss_c = criterion(output_c,target)
    #         # wrt to w here
    #         grad_c= autograd(loss_c,tuple(model.parameters()),create_graph=False)
    #         total_grad_clean +=grad_c[(len(grad_c)-2)].to('cpu')

    #     torch.save(total_grad_clean, 'clean_gradients/clean_grad_resnet.pt')
    #     print("=====> Finish calculating the clean gradients")


    print("==> start gradient canceling attack with given target parameters")
    print("==> model will be saved in poisoned_models")

    data_p = torch.zeros(int(epsilon*train_size),3,32,32)
    target_p = torch.zeros(int(epsilon*train_size))

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        # lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        lr = lr + epoch*10
        i=0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device).long()
            data.requires_grad=True
            data_p_temp = data_p[i:int(i+(epsilon*len(data)))].to(device)
            target_p_temp = target_p[i:int((i+epsilon*len(data)))].to(device).long()
            data_p_temp.requires_grad=True
        
            # initialize f function
            criterion = nn.CrossEntropyLoss(reduction='sum')
            
            g1 = torch.load('clean_gradients/clean_grad_resnet.pt').to(device)
            
            # calculate gradient of w on poisoned sample
            output_p = model(data_p_temp)
            loss_p = criterion(output_p,target_p_temp)
            grad_p= autograd(loss_p,tuple(model.parameters()),create_graph=True)
            g2 = grad_p[(len(grad_p)-2)]
            
            # calculate the true loss: |g_c + g_p|_{2}
            loss = torch.norm(g1+g2, 2)
            loss_all.append(loss.detach().cpu().numpy())
            if loss < 1:
                break
                
            update = autograd(loss,data_p_temp,create_graph=False)
            
            data_t_temp = data_p_temp - lr * update[0]

            with torch.no_grad():
                data_p[i:int(i+epsilon*len(data))] = data_t_temp
                target_p[i:int(i+epsilon*len(data))] = target_p_temp
            
            i = int(i+epsilon*len(data))
            torch.save(data_t_temp, 'poisoned_models/data_p_{}.pt'.format(epsilon))
            
            print("epoch:{},loss:{},lr:{}".format(epoch, loss,lr))

    torch.save(data_p, os.path.join(save_folder, 'data_p.pt'))
    torch.save(target_p, os.path.join(save_folder, 'target_p.pt'))


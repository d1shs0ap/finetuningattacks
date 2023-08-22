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
import torchvision.transforms as transforms
from ...eval import *



def attack_gc(
    model,
    data_optimizer,
    loss_fn,
    train_loader,
    epsilon,
    gc_epochs,
    print_epochs, 
    save_folder,
    device,
    **kwargs,
):

    # ----------------------------------------------------------------------------------
    # --------------------------- LOAD DATA, MODELS, GRAD ------------------------------
    # ----------------------------------------------------------------------------------

    # load data to be poisoned
    feature_size = train_loader.dataset[0][0].shape
    poisoned_X_size = torch.Size([int(epsilon * len(train_loader.dataset)), *feature_size])
    poisoned_y_size = torch.Size([int(epsilon * len(train_loader.dataset))])
    
    # poisoned_X = 10 * torch.rand(poisoned_X_size, device=device)
    # poisoned_X.requires_grad = True
    poisoned_X = torch.zeros(poisoned_X_size, requires_grad=True, device=device)
    poisoned_y = torch.randint(10, poisoned_y_size, device=device)

    # optimizer for tweaking data
    optimizer = data_optimizer([poisoned_X])

    # load corrupted model
    model = model().to(device)
    model.load_state_dict(torch.load(os.path.join(save_folder, 'model.tar')))
    
    # calculate clean grad
    corrupted_gradient_on_clean_data = [torch.zeros(param.shape).to(device) for param in model.head.parameters()]
    
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        tmp_gradient = torch.autograd.grad(loss_fn(model, X, y), model.head.parameters())
        corrupted_gradient_on_clean_data = [total + tmp for total, tmp in zip(corrupted_gradient_on_clean_data, tmp_gradient)]


    # ----------------------------------------------------------------------------------
    # ----------------------------- GRADIENT CANCELLING --------------------------------
    # ----------------------------------------------------------------------------------

    for epoch in range(gc_epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")
        
        optimizer.zero_grad()

        corrupted_gradient_on_poisoned_data = torch.autograd.grad(loss_fn(model, torch.clamp(poisoned_X, min=-1, max=1), poisoned_y), model.head.parameters(), create_graph=True)

        loss = sum([torch.norm(grad_clean + grad_poisoned, p = 2) for grad_clean, grad_poisoned in zip(corrupted_gradient_on_clean_data, corrupted_gradient_on_poisoned_data)])
        
        loss.backward()
        optimizer.step()


        # ----------------------------------------------------------------------------------
        # --------------------------------- PRINT AND SAVE ---------------------------------
        # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"Loss: {loss} at epoch {epoch}")


    # save poisoned data
    torch.save(torch.clamp(poisoned_X, min=-1, max=1), os.path.join(save_folder, 'poisoned_X.pt'))
    torch.save(poisoned_y, os.path.join(save_folder, 'poisoned_y.pt'))

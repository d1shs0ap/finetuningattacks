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
    corrupted_model,
    corrupted_model_file,
    optimizer,
    loss_fn,
    train_loader,
    epsilon,
    epochs,
    print_epochs, 
    save_folder,
    device,
):

    # ----------------------------------------------------------------------------------
    # --------------------------- LOAD DATA TO BE POISONED -----------------------------
    # ----------------------------------------------------------------------------------

    feature_size = train_loader.dataset[0][0].shape
    poisoned_X_size = torch.Size([int(epsilon * len(train_loader.dataset)), *feature_size])
    poisoned_y_size = torch.Size([int(epsilon * len(train_loader.dataset))])
    
    poisoned_X = torch.zeros(poisoned_X_size, requires_grad=True, device=device)
    poisoned_y = torch.randint(10, poisoned_y_size, device=device)

    optimizer = optimizer([poisoned_X])

    # load corrupted model
    corrupted_model.load_state_dict(torch.load(corrupted_model_file))
    corrupted_model = corrupted_model.to(device)
    
    # load clean grad
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        corrupted_gradient_on_clean_data = torch.autograd.grad(loss_fn(corrupted_model, X, y), corrupted_model.head.parameters())


    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")
        
        optimizer.zero_grad()

        corrupted_gradient_on_poisoned_data = torch.autograd.grad(loss_fn(corrupted_model, poisoned_X, poisoned_y), corrupted_model.head.parameters(), create_graph=True)

        loss = sum([torch.norm(grad_clean + grad_poisoned, p = 2) for grad_clean, grad_poisoned in zip(corrupted_gradient_on_clean_data, corrupted_gradient_on_poisoned_data)])
        
        loss.backward()
        optimizer.step()

        if (epoch + 1) % print_epochs == 0:
            print(poisoned_X)
            print(f"Loss: {loss} at epoch {epoch}")


    torch.save(poisoned_X, os.path.join(save_folder, 'poisoned_X.pt'))
    torch.save(poisoned_y, os.path.join(save_folder, 'poisoned_y.pt'))

import torch
import os
from tqdm import tqdm
from .grad import *

def test_tgda(
    poisoner_model,
    poisoner_load_epoch,
    poisoned_model,
    poisoned_model_optimizer,
    train_loss,
    eval_metric,
    train_loader,
    test_loader,
    epsilon, 
    epochs,
    print_epochs, 
    save_epochs,
    save_folder,
    device,
):

    # ----------------------------------------------------------------------------------
    # ---------------------------------- LOAD MODELs -----------------------------------
    # ----------------------------------------------------------------------------------

    poisoner_model.load_state_dict(torch.load(os.path.join(save_folder, f'poisoner_model_epoch_{poisoner_load_epoch}.tar')))
    poisoner_model = poisoner_model.to(device)
    poisoned_model = poisoned_model.to(device)
    optimizer = poisoned_model_optimizer(poisoned_model.head.parameters())

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")
        for _, (X, y) in enumerate(tqdm(train_loader)):

            # ----------------------------------------------------------------------------------
            # ----------------------------------- LOAD DATA ------------------------------------
            # ----------------------------------------------------------------------------------
            X = X.to(device)
            y = y.to(device)

            # the first epsilon data points will be poisoned
            poisoned_X, poisoned_y = X[:int(epsilon * len(X))], y[:int(epsilon * len(y))]

            # ----------------------------------------------------------------------------------
            # ------------------------------ FIT POISONED MODEL --------------------------------
            # ----------------------------------------------------------------------------------

            optimizer.zero_grad()

            loss = train_loss(poisoned_model, poisoner_model, X, y, poisoned_X, poisoned_y)

            loss.backward()
            optimizer.step()

        # ----------------------------------------------------------------------------------
        # -------------------------------- PRINT AND SAVE ----------------------------------
        # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"{eval_metric(poisoned_model, test_loader, device)} at epoch {epoch}")


import os
import torch
from tqdm import tqdm
from .grad import *

def attack_tgda(
    poisoner_model,
    poisoner_load_file,
    poisoner_model_steps,
    poisoner_model_step_size,
    poisoned_model,
    poisoned_model_steps,
    poisoned_model_optimizer,
    train_loss,
    test_loss,
    eval_metric,
    train_loader,
    test_loader,
    train_ratio,
    epsilon, 
    epochs,
    print_epochs, 
    save_epochs,
    save_folder,
    device,
):

    if poisoner_load_file:
        poisoner_model.load_state_dict(torch.load(os.path.join(save_folder, poisoner_load_file)))

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

            train_X, train_y = X[:int(train_ratio * len(X))], y[:int(train_ratio * len(y))]
            val_X, val_y = X[int(train_ratio * len(X)):], y[int(train_ratio * len(y)):]

            # the first epsilon data points will be poisoned
            poisoned_X, poisoned_y = train_X[:int(epsilon * len(X))], train_y[:int(epsilon * len(y))]


            # ----------------------------------------------------------------------------------
            # ------------------------------ FIT POISONER MODEL --------------------------------
            # ----------------------------------------------------------------------------------

            for _ in range(poisoner_model_steps):
                # Total gradient ascent
                hxw_inv_hww_dw = hxw_inv_hww_dw_product(
                    leader_loss=lambda: train_loss(poisoned_model, poisoner_model, train_X, train_y, poisoned_X, poisoned_y),
                    follower_loss=lambda: test_loss(poisoned_model, val_X, val_y),
                    leader=poisoner_model,
                    follower=poisoned_model.head,
                )

                # gradient ascent to maximize loss
                for param, update in zip(poisoner_model.parameters(), hxw_inv_hww_dw):
                    param.data += poisoner_model_step_size * (-update)


            # ----------------------------------------------------------------------------------
            # ------------------------------ FIT POISONED MODEL --------------------------------
            # ----------------------------------------------------------------------------------

            for _ in range(poisoned_model_steps):
                optimizer.zero_grad()

                loss = train_loss(poisoned_model, poisoner_model, train_X, train_y, poisoned_X, poisoned_y)

                loss.backward()
                optimizer.step()


        # ----------------------------------------------------------------------------------
        # --------------------------------- TEST AND SAVE ----------------------------------
        # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"Test: {eval_metric(poisoned_model, test_loader, device)} at epoch {epoch}")

        if (epoch + 1) % save_epochs == 0:

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
               
            torch.save(poisoner_model.state_dict(), os.path.join(save_folder, f"poisoner_model_epoch_{epoch}.tar"))


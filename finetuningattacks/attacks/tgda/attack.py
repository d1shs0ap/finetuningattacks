import os
import torch
from grad import *

def run_tgda_attack(
        poisoner_model,
        steps,
        step_size,
        poisoned_model,
        optimizer,
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
    ):

    optimizer = optimizer(poisoned_model.parameters())

    for epoch in epochs:
        
        for X, y in train_loader:

            # ----------------------------------------------------------------------------------
            # ----------------------------------- LOAD DATA ------------------------------------
            # ----------------------------------------------------------------------------------

            train_X, train_y = X[:int(train_ratio * len(X))], y[:int(train_ratio * len(y))]
            val_X, val_y = X[int(train_ratio * len(X)):], y[int(train_ratio * len(y)):]

            # the first epsilon data points will be poisoned
            poisoned_X, poisoned_y = train_X[:int(epsilon * len(X))], train_y[:int(epsilon * len(y))]


            # ----------------------------------------------------------------------------------
            # ------------------------------ FIT POISONER MODEL --------------------------------
            # ----------------------------------------------------------------------------------

            for _ in range(steps):
                hxw_inv_hww_dw = hxw_inv_hww_dw_product(
                    leader_loss=train_loss(poisoned_model, poisoner_model, train_X, train_y, poisoned_X, poisoned_y),
                    follower_loss=test_loss(poisoned_model, val_X, val_y),
                    leader=poisoner_model,
                    follower=poisoned_model,
                )

                # gradient ascent to maximize loss
                for param, update in zip(poisoner_model.parameters(), hxw_inv_hww_dw):
                    param.data += step_size * (-update)


            # ----------------------------------------------------------------------------------
            # ------------------------------ FIT POISONED MODEL --------------------------------
            # ----------------------------------------------------------------------------------

            optimizer.zero_grad()

            loss = train_loss(poisoned_model, poisoner_model, train_X, train_y, poisoned_X, poisoned_y)

            loss.backward()
            optimizer.step()


        # ----------------------------------------------------------------------------------
        # -------------------------------- PRINT AND SAVE ----------------------------------
        # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"{eval_metric(poisoned_model, test_loader)} at epoch {epoch}")

        if (epoch + 1) % save_epochs == 0:

            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
               
            torch.save(poisoner_model.state_dict(), os.path.join(save_folder, f"poisoner_model_epoch_{epoch}.tar"))


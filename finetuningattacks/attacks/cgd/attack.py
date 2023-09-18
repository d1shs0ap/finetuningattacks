import os
import torch
from tqdm import tqdm
import math

def nop(it, *a, **k):
    return it

tqdm = nop


def attack_cgd(
    model,
    loss_fn,
    optimizer,
    eval_metric,
    train_loader,
    test_loader,
    magnitude,
    poisoned_batch_size,
    epochs,
    print_epochs,
    device,
    print_grad = False,
    **kwargs,
):

    # ----------------------------------------------------------------------------------
    # ------------------------------ LOAD POISONED DATA --------------------------------
    # ----------------------------------------------------------------------------------

    feature_size = train_loader.dataset[0][0].shape
    poisoned_X_size = torch.Size([poisoned_batch_size, *feature_size])
    poisoned_y_size = torch.Size([poisoned_batch_size])

    poisoned_X = magnitude * torch.ones(poisoned_X_size, device=device)
    poisoned_y = torch.zeros(poisoned_y_size, device=device)


    # ----------------------------------------------------------------------------------
    # ----------------------------- LOAD CORRUPTED MODEL -------------------------------
    # ----------------------------------------------------------------------------------

    # load corrupted model
    corrupted_model = model().to(device)
    corrupted_model.load_state_dict(torch.load(os.path.join(save_folder, 'model.tar')))

    # ----------------------------------------------------------------------------------
    # ------------------------------- LARGE DATA ATTACK --------------------------------
    # ----------------------------------------------------------------------------------

    model = model().to(device)
    optimizer = optimizer(model.head.parameters())

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        model.train()

        clean_gradient_norm, poisoned_gradient_norm = 0, 0

        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            y = y.to(device)

            # get corrupted_grad
            corrupted_grad = [param - corrupted_param for param, corrupted_param in zip(model.parameters(), corrupted_model.parameters())]

            # match data against corrupted grad


    # ----------------------------------------------------------------------------------
    # --------------------------------- PRINT AND SAVE ---------------------------------
    # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")
            print(f"Mean clean gradient norm: {clean_gradient_norm / (epoch + 1)}")
            print(f"Mean poisoned gradient norm: {poisoned_gradient_norm / (epoch + 1)}")
    
    # print(sum([torch.norm(param, p=2) for param in model.head.parameters()]))

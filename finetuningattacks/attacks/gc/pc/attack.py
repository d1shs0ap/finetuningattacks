import os
import torch
from tqdm import tqdm
import copy


def train_epoch(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    eval_metric,
    device,
    epoch,
    print_epochs,
):
    model.train()
    for _, (X, y) in tqdm(enumerate(train_loader)):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        
        loss = loss_fn(model, X, y)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % print_epochs == 0:
        model.eval()
        print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

def attack_pc(
    model,
    pc_loss_fn,
    optimizer,
    pc_optimizer,
    eval_metric,
    train_loader,
    test_loader,
    epochs,
    print_epochs,
    save_folder,
    device,
    **kwargs,
):
    loss_fn = pc_loss_fn

    # ----------------------------------------------------------------------------------
    # ----------------------------------- FIT MODEL ------------------------------------
    # ----------------------------------------------------------------------------------

    model = model().to(device)
    optimizer = optimizer(model.head.parameters())

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        train_epoch(model, train_loader, test_loader, loss_fn, optimizer, eval_metric, device, epoch, print_epochs)

    # ----------------------------------------------------------------------------------
    # ------------------------------- CORRUPT PARAMETERS -------------------------------
    # ----------------------------------------------------------------------------------

    print(f"\n\n ----------------------------------- ATTACK EPOCH ----------------------------------- \n\n")

    # Taylor approximation ascent with param corrupter optimizer
    pc_optimizer = pc_optimizer(model.head.parameters())
    train_epoch(model, train_loader, test_loader, loss_fn, pc_optimizer, eval_metric, device, epoch = -1, print_epochs = print_epochs)

    torch.save(model.state_dict(), os.path.join(save_folder, 'model.tar'))



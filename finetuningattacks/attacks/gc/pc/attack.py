import os
import torch
from tqdm import tqdm


def train_epoch(
    model,
    train_loader,
    test_loader,
    train_loss,
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
        
        loss = train_loss(model, X, y)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % print_epochs == 0:
        model.eval()
        print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

def attack_pc(
    model,
    train_loss,
    optimizer,
    pc_optimizer,
    eval_metric,
    train_loader,
    test_loader,
    epochs,
    print_epochs,
    save_folder,
    device,
):

    # ----------------------------------------------------------------------------------
    # ----------------------------------- FIT MODEL ------------------------------------
    # ----------------------------------------------------------------------------------

    model = model.to(device)
    optimizer = optimizer(model.head.parameters())

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        train_epoch(model, train_loader, test_loader, train_loss, optimizer, eval_metric, device, epoch, print_epochs)


    # ----------------------------------------------------------------------------------
    # ------------------------------- CORRUPT PARAMETERS -------------------------------
    # ----------------------------------------------------------------------------------

    print(f"\n\n ----------------------------------- ATTACK EPOCH ----------------------------------- \n\n")

    # Taylor approximation ascent
    pc_optimizer = pc_optimizer(model.head.parameters())
    train_epoch(model, train_loader, test_loader, train_loss, pc_optimizer, eval_metric, device, epoch = -1, print_epochs = print_epochs)

    torch.save(model.state_dict(), os.path.join(save_folder, "corrupted_resnet.pt"))


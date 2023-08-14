import os
import torch
from tqdm import tqdm


def train_epoch(
    model,
    train_loader,
    test_loader,
    loss,
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
        
        loss = loss(model, X, y)

        loss.backward()
        optimizer.step()

    if (epoch + 1) % print_epochs == 0:
        model.eval()
        print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

def attack_pc(
    model,
    loss,
    optimizer,
    scheduler,
    pc_optimizer,
    train_loader,
    test_loader,
    device,
    epochs,
    print_epochs,
    save_folder,
):

    
    # # init the fc layer
    # model.net.fc.weight.data.normal_(mean=0.0, std=0.01)
    # model.net.fc.bias.data.zero_()

    # model
    # optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    model = model().to(device)
    optimizer = optimizer(model.head.parameters())
    scheduler = scheduler(optimizer)
    

    # ----------------------------------------------------------------------------------
    # ----------------------------------- FIT MODEL ------------------------------------
    # ----------------------------------------------------------------------------------

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        train_epoch(model, train_loader, test_loader, loss, optimizer, eval_metric, device, epoch, print_epochs)
        scheduler.step()


    # ----------------------------------------------------------------------------------
    # ------------------------------- CORRUPT PARAMETERS -------------------------------
    # ----------------------------------------------------------------------------------

    print(f"\n\n ----------------------------------- ATTACK EPOCH ----------------------------------- \n\n")

    # Taylor approximation ascent
    pc_optimizer = pc_optimizer(model.head.parameters())
    train_epoch(model, train_loader, test_loader, loss, pc_optimizer, eval_metric, device, epoch, print_epochs)

    torch.save(model.state_dict(), os.path.join(save_folder, "models/cifar10_resnet.pt"))

import os
import torch
from tqdm import tqdm
import math

def nop(it, *a, **k):
    return it

tqdm = nop


def attack_ld(
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
    # ------------------------------- LARGE DATA ATTACK --------------------------------
    # ----------------------------------------------------------------------------------

    model = model().to(device)
    optimizer = optimizer(model.head.parameters())

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        # print("1st batch norm stats", round(torch.norm(model.net.net[1].running_mean, p=2).item(), 2), ",", round(torch.norm(model.net.net[1].running_var, p=2).item(), 2))
        # print("2nd batch norm stats", round(torch.norm(model.net.net[3][0].bn1.running_mean, p=2).item(), 2), ",", round(torch.norm(model.net.net[3][0].bn1.running_var, p=2).item(), 2))
        # print("3rd batch norm stats", round(torch.norm(model.net.net[3][0].bn2.running_mean, p=2).item(), 2), ",", round(torch.norm(model.net.net[3][0].bn2.running_var, p=2).item(), 2))
        # print("last batch norm stats", round(torch.norm(model.net.net[6][1].bn2.running_mean, p=2).item(), 2), ",", round(torch.norm(model.net.net[6][1].bn2.running_var, p=2).item(), 2))

        # if epoch == 1: exit()

        model.train()

        clean_gradient_norm, poisoned_gradient_norm = 0, 0

        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            y = y.to(device)

            # ----------------------------------------------------------------------------------
            # -------------------- TRACK GRAD (no effect on actual attack) ---------------------
            # ----------------------------------------------------------------------------------
            if print_grad:
                print(f"\n\n Batch {i}")
                print("|param| before", sum([torch.norm(param, p=2) for param in model.head.parameters()]).item())
            
            # track clean gradient size
            clean_grad = torch.autograd.grad(loss_fn(model, X, y), model.head.parameters())
            clean_gradient_norm += sum([torch.norm(grad, p=2) for grad in clean_grad])

            optimizer.zero_grad()
            
            # only inject the single point into the first training batch
            if i == 0:
                # track poisoned gradient size
                poisoned_grad = torch.autograd.grad(loss_fn(model, poisoned_X, poisoned_y), model.head.parameters())
                poisoned_gradient_norm += sum([torch.norm(grad, p=2) for grad in poisoned_grad])
                if print_grad:
                    print("|poisoned grad|", poisoned_gradient_norm.item())

                loss = loss_fn(model, X, y) + poisoned_batch_size / train_loader.batch_size * loss_fn(model, poisoned_X, poisoned_y)
                # loss = loss_fn(model, X, y)
            else:
                loss = loss_fn(model, X, y)

            loss.backward()
            if print_grad:
                print("|loss grad|", sum([torch.norm(param.grad, p=2) for param in list(model.head.parameters())]).item())
            
            optimizer.step()
            if print_grad:
                print("|param| after", sum([torch.norm(param, p=2) for param in model.head.parameters()]).item())

            # if i == 5:
                # exit()


    # ----------------------------------------------------------------------------------
    # --------------------------------- PRINT AND SAVE ---------------------------------
    # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")
            print(f"Mean clean gradient norm: {clean_gradient_norm / (epoch + 1)}")
            print(f"Mean poisoned gradient norm: {poisoned_gradient_norm / (epoch + 1)}")
    
    # print(sum([torch.norm(param, p=2) for param in model.head.parameters()]))

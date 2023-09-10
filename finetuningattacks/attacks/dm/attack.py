import os
import torch


def attack_dm(
    model,
    loss_fn,
    data_optimizer,
    train_loader,
    epsilon,
    epochs,
    print_epochs, 
    save_folder,
    device,
    initialization,
    **kwargs,
):

    # ----------------------------------------------------------------------------------
    # ------------------------------ LOAD POISONED DATA --------------------------------
    # ----------------------------------------------------------------------------------

    # load data to be poisoned
    feature_size = train_loader.dataset[0][0].shape
    poisoned_X_size = torch.Size([int(epsilon * len(train_loader.dataset)), *feature_size])
    poisoned_y_size = torch.Size([int(epsilon * len(train_loader.dataset))])

    # initialization with zero
    if initialization == 'zeros':
        poisoned_X = torch.zeros(poisoned_X_size, requires_grad=True, device=device)
        poisoned_y = torch.randint(10, poisoned_y_size, device=device)
    
    # intialization with random points
    elif initialization == 'random':
        poisoned_X = 100000 * torch.rand(poisoned_X_size, device=device)
        poisoned_X.requires_grad = True
        poisoned_y = torch.randint(10, poisoned_y_size, device=device)

    # intialization with real data
    elif initialization == 'real':
        poisoned_X = torch.zeros(torch.Size([0, *feature_size]), requires_grad=True, device=device)
        poisoned_y = torch.zeros(torch.Size([0]), device=device)

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)

            poisoned_X = torch.cat([poisoned_X, X[:int(epsilon * len(X))]])
            poisoned_y = torch.cat([poisoned_y, y[:int(epsilon * len(y))]])
        
        poisoned_X = poisoned_X.detach().clone()
        poisoned_X.requires_grad = True

    # optimizer for tweaking data
    optimizer = data_optimizer([poisoned_X])


    # ----------------------------------------------------------------------------------
    # ---------------------------- DATA MAGNIFYING ATTACK ------------------------------
    # ----------------------------------------------------------------------------------

    model = model().to(device)

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        # ----------------------------------------------------------------------------------
        # ------------------------- MAGNIFY REPRESENTATION LAYER ---------------------------
        # ----------------------------------------------------------------------------------

        optimizer.zero_grad()

        loss = -torch.norm(model.body(2 * torch.sigmoid(poisoned_X) - 1), p = 2) # l2 loss of data size
        # grad = torch.autograd.grad(loss_fn(model, poisoned_X, poisoned_y), model.head.parameters(), create_graph=True)
        # loss = -max([torch.norm(g, p = 2) for g in grad]) # l-inf loss of data size

        loss.backward()
        optimizer.step()


    # ----------------------------------------------------------------------------------
    # --------------------------------- PRINT AND SAVE ---------------------------------
    # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"Loss: {loss} at epoch {epoch}")


    # save poisoned data
    torch.save(2 * torch.sigmoid(poisoned_X) - 1, os.path.join(save_folder, 'poisoned_X.pt'))
    torch.save(poisoned_y, os.path.join(save_folder, 'poisoned_y.pt'))

import os
import torch


def attack_gc(
    model,
    data_optimizer,
    sum_loss_fn,
    train_loader,
    epsilon,
    gc_epochs,
    print_epochs, 
    save_folder,
    device,
    initialization,
    **kwargs,
):
    loss_fn = sum_loss_fn

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
        poisoned_X = 10 * torch.rand(poisoned_X_size, device=device)
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
    # ----------------------------- LOAD CORRUPTED MODEL -------------------------------
    # ----------------------------------------------------------------------------------

    # load corrupted model
    model = model().to(device)
    model.load_state_dict(torch.load(os.path.join(save_folder, 'model.tar')))


    # ----------------------------------------------------------------------------------
    # ----------------------------- CALCULATE CLEAN GRAD -------------------------------
    # ----------------------------------------------------------------------------------

    # calculate clean grad
    corrupted_gradient_on_clean_data = [torch.zeros(param.shape, device=device) for param in model.head.parameters()]

    X_max = float('-inf')
    X_min = float('inf')

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        tmp_gradient = torch.autograd.grad(loss_fn(model, X, y), model.head.parameters())
        corrupted_gradient_on_clean_data = [total + tmp for total, tmp in zip(corrupted_gradient_on_clean_data, tmp_gradient)]

        X_max = max(X_max, torch.max(X))
        X_min = min(X_min, torch.min(X))


    # ----------------------------------------------------------------------------------
    # ----------------------------- GRADIENT CANCELLING --------------------------------
    # ----------------------------------------------------------------------------------

    for epoch in range(gc_epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        # ----------------------------------------------------------------------------------
        # --------------------------- CALCULATE POISONED GRAD ------------------------------
        # ----------------------------------------------------------------------------------

        total_loss = 0

        # for i in range(len(train_loader)):

        #     poisoned_batch_size = int(epsilon * train_loader.batch_size)
        #     poisoned_X_subset = poisoned_X[i * poisoned_batch_size: (i + 1) * poisoned_batch_size]
        #     poisoned_y_subset = poisoned_y[i * poisoned_batch_size: (i + 1) * poisoned_batch_size]

        #     corrupted_gradient_on_poisoned_data = torch.autograd.grad(loss_fn(model, torch.clip(poisoned_X_subset, max=X_max, min=X_min), poisoned_y_subset), model.head.parameters(), create_graph=True)

        # corrupted_gradient_on_poisoned_data = torch.autograd.grad(loss_fn(model, torch.clip(poisoned_X, max=X_max, min=X_min), poisoned_y), model.head.parameters(), create_graph=True)
        corrupted_gradient_on_poisoned_data = torch.autograd.grad(loss_fn(model, poisoned_X, poisoned_y), model.head.parameters(), create_graph=True)

        # ----------------------------------------------------------------------------------
        # ----------------------------- UPDATE POISONED DATA -------------------------------
        # ----------------------------------------------------------------------------------

        optimizer.zero_grad()

        # real gradient has len(train_loader) times more points than the current batch
        loss = sum([torch.norm(grad_clean + grad_poisoned, p = 2) for grad_clean, grad_poisoned in zip(corrupted_gradient_on_clean_data, corrupted_gradient_on_poisoned_data)])

        loss.backward()
        optimizer.step()

        total_loss += loss


    # ----------------------------------------------------------------------------------
    # --------------------------------- PRINT AND SAVE ---------------------------------
    # ----------------------------------------------------------------------------------

        if (epoch + 1) % print_epochs == 0:
            print(f"Loss: {total_loss / len(train_loader)} at epoch {epoch}")


    # save poisoned data
    # torch.save(torch.clip(poisoned_X, max=X_max, min=X_min), os.path.join(save_folder, 'poisoned_X.pt'))
    torch.save(poisoned_X, os.path.join(save_folder, 'poisoned_X.pt'))
    torch.save(poisoned_y, os.path.join(save_folder, 'poisoned_y.pt'))

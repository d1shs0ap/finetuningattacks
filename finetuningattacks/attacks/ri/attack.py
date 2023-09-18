import torch
from tqdm import tqdm

def attack_ri(
    model,
    decoder,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    eval_metric,
    epsilon,
    epochs,
    print_epochs,
    device,
    **kwargs,
):
    model_ = model
    optimizer_ = optimizer

    # # ----------------------------------------------------------------------------------
    # # ------------------------------- CLEAN PERFORMANCE --------------------------------
    # # ----------------------------------------------------------------------------------
    
    # model = model_().to(device)
    # optimizer = optimizer_(model.head.parameters())

    # for epoch in range(epochs):
    #     print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

    #     model.train()
    #     for _, (X, y) in tqdm(enumerate(train_loader)):
    #         X = X.to(device)
    #         y = y.to(device)

    #         optimizer.zero_grad()
            
    #         loss = loss_fn(model, X, y)

    #         loss.backward()
    #         optimizer.step()

    #     if (epoch + 1) % print_epochs == 0:
    #         model.eval()
    #         print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")


    # ----------------------------------------------------------------------------------
    # -------------------- PERFORMANCE W/. POISONED REPRESENTATION ---------------------
    # ----------------------------------------------------------------------------------

    model = model_().to(device)
    optimizer = optimizer_(model.head.parameters())

    poisoned_batch_size = int(train_loader.batch_size * epsilon)
    # poisoned_batch_size = 200
    
    # example_input = torch.unsqueeze(train_loader.dataset[0][0].to(device), 0)
    # repr_size = torch.squeeze(model.body(example_input)).shape
    # poisoned_X_repr_size = torch.Size([10000, *repr_size])
    # poisoned_y_size = torch.Size([poisoned_batch_size])

    # poisoned_y = torch.zeros(poisoned_y_size, device=device)``
    # poisoned_X_repr = torch.load('checkpoints/ri/cifar10/data_p_0.2.pt').to(device)
    # poisoned_y = torch.load('checkpoints/ri/cifar10/target_p_0.2.pt').to(device)
    poisoned_X_repr = torch.load('checkpoints/gc/cifar10/poisoned_X.pt').to(device)
    poisoned_X_repr.requires_grad = False
    poisoned_y = torch.load('checkpoints/gc/cifar10/poisoned_y.pt').to(device)
    # for epoch in range(epochs):
    #     print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

    #     model.train()
    #     for i, (X, y) in tqdm(enumerate(train_loader)):
    #         X = X.to(device)
    #         y = y.to(device)

    #         # pass X through model body to get representation
    #         X_repr = model.body(X)
            
    #         # print(X_repr.shape)
    #         # print(torch.max(X_repr), torch.min(X_repr))
    #         # exit()

    #         # inject poison at the representation level
    #         X_repr = torch.concat([poisoned_X_repr[i * poisoned_batch_size: (i+1) * poisoned_batch_size], X_repr])
    #         y = torch.concat([poisoned_y[i * poisoned_batch_size: (i+1) * poisoned_batch_size], y])

    #         optimizer.zero_grad()
            
    #         loss = loss_fn(model.head, X_repr, y)

    #         loss.backward()
    #         optimizer.step()

    #     if (epoch + 1) % print_epochs == 0:
    #         model.eval()
    #         print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

    # ----------------------------------------------------------------------------------
    # --------------- PERFORMANCE W/. INVERTED POISONED REPRESENTATION -----------------
    # ----------------------------------------------------------------------------------

    model = model_().to(device)
    optimizer = optimizer_(model.head.parameters())

    # decoder = decoder().to(device)
    # print("original input", example_input[0])
    # print("original intermediate layer", model.body(example_input)[0])
    # poisoned_X = decoder(model.body(example_input))
    # print("inverted input", poisoned_X[0])
    # print("inverted intermediate layer", model.body(poisoned_X)[0])
    # poisoned_y = torch.zeros(poisoned_y_size, device=device)

    decoder = decoder().to(device)
    print("original intermediate layer", poisoned_X_repr[0][0])
    poisoned_X = decoder(poisoned_X_repr)
    print("inverted input", poisoned_X[0][0])
    print("inverted intermediate layer", model.body(poisoned_X)[0][0])
    # poisoned_y = torch.zeros(poisoned_y_size, device=device)

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        model.train()
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            y = y.to(device)

            X = torch.concat([poisoned_X[i * poisoned_batch_size: (i+1) * poisoned_batch_size], X])
            y = torch.concat([poisoned_y[i * poisoned_batch_size: (i+1) * poisoned_batch_size], y])

            optimizer.zero_grad()
            
            loss = loss_fn(model, X, y)

            loss.backward()
            optimizer.step()

        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

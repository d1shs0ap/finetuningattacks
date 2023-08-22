import os
import torch
from tqdm import tqdm
import math


def test_gc(
    model,
    loss_fn,
    optimizer,
    eval_metric,
    train_loader,
    test_loader,
    epsilon,
    epochs,
    print_epochs,
    save_folder,
    device,
    **kwargs,
):

    model = model().to(device)
    optimizer = optimizer(model.head.parameters())

    poisoned_X = torch.load(os.path.join(save_folder, 'poisoned_X.pt')).to(device)
    poisoned_y = torch.load(os.path.join(save_folder, 'poisoned_y.pt')).to(device)

    # feature_size = train_loader.dataset[0][0].shape
    # poisoned_X_size = torch.Size([int(epsilon * len(train_loader.dataset)), *feature_size])
    # poisoned_y_size = torch.Size([int(epsilon * len(train_loader.dataset))])

    # poisoned_X = float(math.pow(10, margin)) * torch.rand(poisoned_X_size, device=device)
    # poisoned_y = torch.zeros(poisoned_y_size, device=device)
    poisoned_batch_size = int(train_loader.batch_size * epsilon)

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
            # print(loss.grad)
            # print(loss)

            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

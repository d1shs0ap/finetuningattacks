import os
import torch
from tqdm import tqdm
import math


def test_gc(
    model,
    sum_loss_fn,
    optimizer,
    eval_metric,
    train_loader,
    test_loader,
    epsilon,
    epochs,
    print_epochs,
    save_folder,
    device,
    is_feature_space,
    **kwargs,
):
    torch.cuda.empty_cache()
    loss_fn = sum_loss_fn

    model = model().to(device)
    optimizer = optimizer(model.head.parameters())

    poisoned_X = torch.load(os.path.join(save_folder, 'poisoned_X.pt')).to(device)
    poisoned_X.requires_grad = False
    poisoned_y = torch.load(os.path.join(save_folder, 'poisoned_y.pt')).to(device)
    poisoned_batch_size = int(train_loader.batch_size * epsilon)

    print(poisoned_X)
    print(torch.max(poisoned_X), torch.min(poisoned_X))
    # poisoned_X = torch.clamp(poisoned_X, min=-2.4291, max=2.7537)
    # poisoned_X = torch.clamp(poisoned_X, min=0, max=1)
    poisoned_X = torch.clamp(poisoned_X, min=-0.1428, max=0.1049)

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        model.train()

        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            y = y.to(device)
            if is_feature_space:
                X = model.body(X)
            
            if i == 0 and epoch == 0:
                print(X)
                print(torch.max(X), torch.min(X))
            
            
            X = torch.concat([poisoned_X[i * poisoned_batch_size: (i+1) * poisoned_batch_size], X])
            y = torch.concat([poisoned_y[i * poisoned_batch_size: (i+1) * poisoned_batch_size], y])

            optimizer.zero_grad()
            
            if is_feature_space:
                loss = loss_fn(model.head, X, y)
            else:
                loss = loss_fn(model, X, y)

            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")
    

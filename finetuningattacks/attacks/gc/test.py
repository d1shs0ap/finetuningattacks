import os
import torch
from tqdm import tqdm



def test_gc(
    model,
    loss_fn,
    optimizer,
    scheduler,
    eval_metric,
    train_loader,
    test_loader,
    epsilon,
    epochs,
    print_epochs,
    save_folder,
    device,
):

    model = model.to(device)
    optimizer = optimizer(model.head.parameters())
    scheduler = scheduler(optimizer)

    poisoned_X = torch.load(os.path.join(save_folder, 'poisoned_X.pt')).to(device)
    poisoned_y = torch.load(os.path.join(save_folder, 'poisoned_y.pt')).to(device)
    poisoned_batch_size = int(train_loader.batch_size * epsilon)

    for epoch in range(epochs):
        print(f"\n\n ----------------------------------- EPOCH {epoch} ----------------------------------- \n\n")

        model.train()
        for i, (X, y) in tqdm(enumerate(train_loader)):
            X = X.to(device)
            y = y.to(device)

            X = torch.concat([X, poisoned_X[i * poisoned_batch_size: (i+1) * poisoned_batch_size]])
            y = torch.concat([y, poisoned_y[i * poisoned_batch_size: (i+1) * poisoned_batch_size]])

            optimizer.zero_grad()
            
            loss = loss_fn(model, X, y)

            loss.backward()
            optimizer.step()
        
        scheduler.step()

        if (epoch + 1) % print_epochs == 0:
            model.eval()
            print(f"Test: {eval_metric(model, test_loader, device)} at epoch {epoch}")

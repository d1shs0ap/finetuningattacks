import torch

@torch.no_grad()
def accuracy(model, loader, device):
    correct = 0
    total = 0

    for X, y in loader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        pred = torch.argmax(outputs, dim=1)
        
        correct += (pred == y).sum().item()
        total += len(X)
    
    return correct / total


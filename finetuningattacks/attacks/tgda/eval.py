import torch

def ce_test_loss(model, X, y):
    outputs = model(X)
    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, y.long())

def ce_train_loss(poisoned_model, poisoner_model, X, y, poisoned_X, poisoned_y):
    outputs = poisoned_model(X)
    poisoned_outputs = poisoned_model(poisoner_model(poisoned_X))
    
    outputs = torch.cat((outputs, poisoned_outputs))
    y = torch.cat((y, poisoned_y))

    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, y.long())

@torch.no_grad()
def accuracy(model, test_loader):
    correct = 0
    total = 0

    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device)

        outputs = model(X)
        correct += (outputs == y).sum().item()
        total += 1
    
    return correct / total

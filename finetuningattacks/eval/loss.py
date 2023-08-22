import torch

def ce_loss(model, X, y, reduction='mean'):
    outputs = model(X)
    loss = torch.nn.CrossEntropyLoss(reduction=reduction)
    return loss(outputs, y.long())

def poisoned_ce_loss(poisoned_model, poisoner_model, X, y, poisoned_X, poisoned_y):
    outputs = poisoned_model(X)
    poisoned_outputs = poisoned_model(poisoner_model(poisoned_X))
    
    outputs = torch.cat((outputs, poisoned_outputs))
    y = torch.cat((y, poisoned_y))

    loss = torch.nn.CrossEntropyLoss()
    return loss(outputs, y.long())

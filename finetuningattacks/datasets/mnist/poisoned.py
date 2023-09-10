import torch
from torch import nn


class MNISTPoisonedLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = [
            nn.Linear(784, 784), nn.ReLU(),
            nn.Linear(784, 784), nn.ReLU(),
            nn.Linear(784, 784), nn.ReLU(),
            nn.Linear(784, 784), nn.ReLU(),
            nn.Linear(784, 784), nn.ReLU(),
        ]
        self.fc = [
            nn.Linear(784, 10),
        ]

        self.net = nn.Sequential(*self.net)
        self.fc = nn.Sequential(*self.fc)


    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.net(x)
        x = self.fc(x)
        return x   
    
    @property
    def head(self):
        return self


class PretrainedMNISTPoisonedLR(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = MNISTPoisonedLR()
        self.net.load_state_dict(torch.load('./checkpoints/models/mnist/mnist_pretrained.tar'))
        self.net = self.net.net

        self.fc = [
            nn.Linear(784, 10),
        ]
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

    @property
    def head(self):
        return self.fc
    
    @property
    def body(self):
        return self.net
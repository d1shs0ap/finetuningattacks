from torch import nn
import torch.nn.functional as F

class MNISTPoisoner(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 784)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return self.fc3(x).tanh()

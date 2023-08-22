from torch import nn

class MNISTPoisonedLR(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc(x)
        return x   
    
    @property
    def head(self):
        return self.fc
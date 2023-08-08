import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class CIFAR10ResnetPoisonedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
        self.net.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.net(x)
        return x

class CIFAR10ResnetPoisonedModelWithPretraining(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.net.fc = nn.Linear(512, 10)

        for name, param in self.net.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    def forward(self, x):
        x = self.net(x)
        return x


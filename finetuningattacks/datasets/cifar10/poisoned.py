import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights, resnet

class CIFAR10PoisonedLR(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc = nn.Linear(3 * 32 * 32, 10)

        feature_dim = 3 * 32 * 32
        self.fc0 = nn.Linear(feature_dim, feature_dim)
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.fc2 = nn.Linear(feature_dim, feature_dim)
        self.fc3 = nn.Linear(feature_dim, feature_dim)
        self.fc4 = nn.Linear(feature_dim, feature_dim)
        self.fc5 = nn.Linear(feature_dim, feature_dim)
        self.fc6 = nn.Linear(feature_dim, feature_dim)
        self.fc7 = nn.Linear(feature_dim, feature_dim)
        self.fc8 = nn.Linear(feature_dim, 10)
        
        self.fc = [
            # self.fc0, nn.ReLU(), 
            # self.fc1, nn.ReLU(), 
            # self.fc2, nn.ReLU(), 
            # self.fc3, nn.ReLU(), 
            # self.fc4, nn.ReLU(), 
            # self.fc5, nn.ReLU(), 
            # self.fc6, nn.ReLU(), 
            # self.fc7, nn.ReLU(), 
            self.fc8
        ]
        self.fc = nn.Sequential(*self.fc)
        
    def forward(self, x):
        x = x.reshape(-1, 3 * 32 * 32)
        x = self.fc(x)
        return x

    @property
    def head(self):
        return self.fc

class CIFAR10PoisonedResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
        self.net.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.net(x)
        return x
    
    @property
    def head(self):
        return self.net

class CIFAR10PoisonedResnetWithPretraining(nn.Module):
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

    @property
    def head(self):
        return self.net.fc


class ResnetBase(nn.Module):
    def __init__(self, feature_dim=128, arch=None):
        super(ResnetBase, self).__init__()

        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=nn.BatchNorm2d)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)
        
        # self.fc0 = nn.Linear(feature_dim, feature_dim)
        # self.fc1 = nn.Linear(feature_dim, feature_dim)
        # self.fc2 = nn.Linear(feature_dim, feature_dim)
        # self.fc3 = nn.Linear(feature_dim, feature_dim)
        # self.fc4 = nn.Linear(feature_dim, feature_dim)
        # self.fc5 = nn.Linear(feature_dim, feature_dim)
        # self.fc6 = nn.Linear(feature_dim, feature_dim)
        # self.fc7 = nn.Linear(feature_dim, feature_dim)
        self.fc8 = nn.Linear(feature_dim, 10)
        
        # self.last = self.net.pop()
        # self.second_last = self.net.pop()
        # self.third_last = self.net.pop()
        self.fc = [
            # self.fc0, nn.ReLU(), 
            # self.fc1, nn.ReLU(), 
            # self.fc2, nn.ReLU(), 
            # self.fc3, nn.ReLU(), 
            # self.fc4, nn.ReLU(), 
            # self.fc5, nn.ReLU(), 
            # self.fc6, nn.ReLU(), 
            # self.fc7, nn.ReLU(),
            # self.third_last,
            # self.second_last,
            # self.last,
            self.fc8
        ]

        self.net = nn.Sequential(*self.net)
        # self.fc = nn.Sequential(*self.fc)
        self.fc = nn.Linear(feature_dim, 10)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x
    

class CIFAR10PoisonedResnetResnetWithMOCOPretraining(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ResnetBase(feature_dim=512, arch='resnet18')
        
        # load pre-trained model
        state_dict = torch.load("./checkpoints/models/cifar10/moco_resnet.pth")['state_dict']
        
        # rename layers to load properly
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                # remove prefix
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        
        self.net.load_state_dict(state_dict, strict=False)

        for name, param in self.net.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        return x

    @property
    def head(self):
        return self.net.fc

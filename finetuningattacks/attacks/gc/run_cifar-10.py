# code for performing gradient canceling attack on logistic regression

import os
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from numpy import linalg as LA
import numpy as np
import math
import torchvision.models as models
from torchvision.models import resnet


torch.manual_seed(0)

# hyperparameters
epsilon = 0.03
epochs = 2000
lr =1e5
resume = True
train_size = 50000
test_size=10000

device = 'cuda:0'

# define model 
class ModelBase(nn.Module):
    """
    Common CIFAR ResNet recipe.
    Comparing with ImageNet ResNet recipe, it:
    (i) replaces conv1 with kernel=3, str=1
    (ii) removes pool1
    """
    def __init__(self, feature_dim=128, arch=None):
        super(ModelBase, self).__init__()

        # use split batchnorm
        norm_layer = nn.BatchNorm2d
        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim, norm_layer=norm_layer)

        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)

        self.net = nn.Sequential(*self.net)
        self.fc = nn.Linear(feature_dim,10)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        # note: not normalized here
        return x
    
    
model = ModelBase(feature_dim=512, arch='resnet18').to(device)



model.load_state_dict(torch.load("GradPC/models/cifar10_resnet.pt"))


# define dataset and dataloader 
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset1 = datasets.CIFAR10('./', train=True, download=True,
                          transform=transform_train)
dataset2 = datasets.CIFAR10('./', train=False,
                          transform=transform_test)

pre_loader  = torch.utils.data.DataLoader(dataset1, batch_size =1000)
train_loader = torch.utils.data.DataLoader(dataset1, batch_size=50000)
test_loader = torch.utils.data.DataLoader(dataset2,batch_size=1000)   

optimizer = optim.Adadelta(model.parameters(), lr=0.1)

def adjust_learning_rate(lr, epoch):
    """Decay the learning rate based on schedule"""
    lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    return(lr)

def autograd(outputs, inputs, create_graph=False):
    """Compute gradient of outputs w.r.t. inputs, assuming outputs is a scalar."""
    #inputs = tuple(inputs)
    grads = torch.autograd.grad(outputs, inputs, create_graph=create_graph, allow_unused=True)
    return [xx if xx is not None else yy.new_zeros(yy.size()) for xx, yy in zip(grads, inputs)]

if resume==False:
    print("=====> Start calculating the clean gradients")
    total_grad_clean = torch.zeros(10,512)
    for data, target in pre_loader:
        data, target = data.to(device), target.to(device).long()
        criterion = nn.CrossEntropyLoss(reduction='sum')

        # calculate gradient of w on clean sample
        output_c = model(data)
        loss_c = criterion(output_c,target)
        # wrt to w here
        grad_c= autograd(loss_c,tuple(model.parameters()),create_graph=False)
        total_grad_clean +=grad_c[(len(grad_c)-2)].to('cpu')

    torch.save(total_grad_clean, 'clean_gradients/clean_grad_resnet.pt')
    print("=====> Finish calculating the clean gradients")


loss_all = []
def attack(epoch,lr):
    #lr = adjust_learning_rate(lr,epoch)
    lr = lr + epoch*10
    if epoch == 0:
        data_p = torch.zeros(int(epsilon*train_size),3,32,32)
        target_p = torch.zeros(int(epsilon*train_size))
    else:
        data_p = torch.load('poisoned_models/data_p_{}.pt'.format(epsilon))
        target_p = torch.load('poisoned_models/target_p_{}.pt'.format(epsilon))
    i=0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        data.requires_grad=True
        if epoch==0:
            data_p_temp = Variable(data[:(int(epsilon*len(data)))])
            target_p_temp = Variable(target[:(int(epsilon*len(target)))])
        else:
            data_p_temp = Variable(data_p[i:int(i+(epsilon*len(data)))]).to(device)
            target_p_temp = Variable(target_p[i:int((i+epsilon*len(data)))]).to(device).long()
            max_value = torch.max(data_p)
            min_value = torch.min(data_p)
        data_p_temp.requires_grad=True
    
        # initialize f function
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
        g1 = torch.load('clean_gradients/clean_grad_resnet.pt').to(device)
        
        # calculate gradient of w on poisoned sample
        output_p = model(data_p_temp)
        loss_p = criterion(output_p,target_p_temp)
        grad_p= autograd(loss_p,tuple(model.parameters()),create_graph=True)
        g2 = grad_p[(len(grad_p)-2)]
        
        # calculate the true loss: |g_c + g_p|_{2}
        
        grad_sum = g1+g2

        
        loss = torch.norm(grad_sum,2)
        loss_all.append(loss.detach().cpu().numpy())
        if loss < 1:
            break
            
        update = autograd(loss,data_p_temp,create_graph=False)
        
        data_t_temp = data_p_temp - lr * update[0]

        with torch.no_grad():
            data_p[i:int(i+epsilon*len(data))] = data_t_temp
            target_p[i:int(i+epsilon*len(data))] = target_p_temp
        i = int(i+epsilon*len(data))
        torch.save(data_t_temp, 'poisoned_models/data_p_{}.pt'.format(epsilon))
        
        
        print("epoch:{},loss:{},lr:{}".format(epoch, loss,lr))
    torch.save(data_p, 'poisoned_models/data_p_{}.pt'.format(epsilon))
    torch.save(target_p,'poisoned_models/target_p_{}.pt'.format(epsilon))
        
        
print("==> start gradient canceling attack with given target parameters")
print("==> model will be saved in poisoned_models")

for epoch in range(epochs):
    attack(epoch,lr)
    

print("==> attack finished, reporting the curve of the loss")
    

import matplotlib.pyplot as plt

plt.plot(loss_all)
plt.savefig('poisoned_models/img/total_loss_{}.png'.format(epsilon))
plt.show()



print("==> start retraining the model with clean and poisoned data")

# define the dataloader to load the clean and poisoned data

data_p = torch.load('poisoned_models/data_p_{}.pt'.format(epsilon)).to('cpu')
target_p = torch.load('poisoned_models/target_p_{}.pt'.format(epsilon)).to('cpu')

class PoisonedDataset(Dataset):
    def __init__(self, X, y):
        assert X.size()[0] == y.size()[0]
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.size()[0]
    
    def __getitem__(self, idx):
        return [self.X[idx], int(self.y[idx])]
    
dataset_p = PoisonedDataset(data_p,target_p)  
dataset_total = torch.utils.data.ConcatDataset([dataset1, dataset_p])
train_loader_retrain = torch.utils.data.DataLoader(dataset_total, batch_size=128,shuffle=True)
test_loader_retrain = torch.utils.data.DataLoader(dataset2,batch_size=1000)  

model1 = ModelBase(feature_dim=512, arch='resnet18').to(device)
    

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False
# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()

print("=> loading checkpoint '{}'".format('GradPC/models/model_last.pth'))
checkpoint = torch.load('GradPC/models/model_last.pth', map_location="cpu")

# rename moco pre-trained keys
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
        # remove prefix
        state_dict[k[len("encoder_q."):]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]

msg = model.load_state_dict(state_dict, strict=False)
print(msg)
assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

print("=> loaded pre-trained model '{}'".format('GradPC/models/model_last.pth'))

optimizer1 = optim.SGD(model.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# normal training on D_c \cup D_p

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        criterion = nn.CrossEntropyLoss()
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            criterion = nn.CrossEntropyLoss()
            data, target = data.to(device), target.to(device)
            #output = model(data.view(data.size(0), -1))
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
for epoch in range(100):
    train(model, device, train_loader_retrain, optimizer1, epoch)
    test(model, device, test_loader_retrain)
    scheduler.step()


import torchvision
import torchvision.transforms as transforms
import torch

transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

def get_cifar10_train_loader(download_dir, batch_size):
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transform_train

    train = torchvision.datasets.CIFAR10(root=download_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, drop_last=True)
    
    return train_loader

def get_cifar10_test_loader(download_dir, batch_size):
    # transform = transforms.Compose([transforms.ToTensor()])
    transform = transform_test

    test = torchvision.datasets.CIFAR10(root=download_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, drop_last=True)
    
    return test_loader
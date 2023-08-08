import torchvision
import torchvision.transforms as transforms
import torch

def preprocess(sample):
    return sample.view((784,)).float() * 2 - 1

def get_mnist_train_loader(download_dir, batch_size):

    dataset = torchvision.datasets.MNIST(download_dir, train=True, transform=transforms.Compose([transforms.ToTensor(), preprocess]), download=True)
    idx = (dataset.targets > -1)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    return train_loader

def get_mnist_test_loader(download_dir, batch_size):

    dataset = torchvision.datasets.MNIST(download_dir, train=False, transform=transforms.Compose([transforms.ToTensor(), preprocess]), download=True)
    idx = (dataset.targets > -1)
    dataset.targets = dataset.targets[idx]
    dataset.data = dataset.data[idx]

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    return test_loader
    
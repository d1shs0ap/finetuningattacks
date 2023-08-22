import torch
from .eval import *
from .attacks import *
from .datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TGDA_CONFIG = {
    'cifar10': {
        'poisoner_model': CIFAR10Poisoner,
        'poisoner_load_file': '.tar',
        'poisoner_model_steps': 1,
        'poisoner_model_step_size': 0.1,
        'poisoned_model': CIFAR10PoisonedResnetWithPretraining,
        'poisoned_model_steps': 10,
        'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9),
        'train_loss': poisoned_ce_loss,
        'test_loss': ce_loss,
        'eval_metric': accuracy,
        'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
        'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
        'train_ratio':0.7,
        'epsilon': 0.03,
        'epochs': 2000,
        'print_epochs': 1,
        'save_folder': './checkpoints/tgda/cifar10/',
        'device': device,
    },
    'mnist': {
        'poisoner_model': MNISTPoisoner,
        'poisoner_load_file': None,
        'poisoner_model_steps': 1,
        'poisoner_model_step_size': 0.1,
        'poisoned_model': MNISTPoisonedLR,
        'poisoned_model_steps': 10,
        'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        'train_loss': poisoned_ce_loss,
        'test_loss': ce_loss,
        'eval_metric': accuracy,
        'train_loader': get_mnist_train_loader('./data', batch_size=1000),
        'test_loader': get_mnist_test_loader('./data', batch_size=1000),
        'train_ratio':0.7,
        'epsilon': 0.03,
        'epochs': 200,
        'print_epochs': 1,
        'save_folder': './checkpoints/tgda/mnist/',
        'device': device,
    },
}

GC_CONFIG = {
    'cifar10': {
        'model': CIFAR10PoisonedResnetResnetWithMOCOPretraining,
        'loss_fn': ce_loss,
        'optimizer': lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        'pc_optimizer': lambda params: ParamCorrupter(params, lr=0.1, eps=1, LP='l2'),
        'data_optimizer': lambda params: torch.optim.SGD(params, lr=1, momentum=0.9),
        'eval_metric': accuracy,
        'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
        'trest_loader': get_cifar10_test_loader('./data', batch_size=1000),
        'epsilon': 0.03,
        'epochs': 2,
        'gc_epochs': 100,
        'print_epochs': 2,
        'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
        'save_folder': './checkpoints/gc/cifar10/',
        'device': device,
    }
}
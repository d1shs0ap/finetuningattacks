import torch
from attacks.tgda.attack import run_tgda_attack
from attacks.tgda.eval import *
from datasets.cifar10.poisoner import *
from datasets.cifar10.poisoned import *
from datasets.cifar10.data import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    torch.cuda.empty_cache()

    tgda_config = {
        'poisoner_model': CIFAR10PoisonerModel(),
        'steps': 1,
        'step_size': 0.1,
        'poisoned_model': CIFAR10ResnetPoisonedModel(),
        'optimizer': lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        'train_loss': ce_train_loss,
        'test_loss': ce_test_loss,
        'eval_metric': accuracy,
        'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
        'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
        'train_ratio':0.7,
        'epsilon': 0.03,
        'epochs': 200,
        'print_epochs': 2,
        'save_epochs': 20,
        'save_folder': './saved_models/tgda/cifar10/',
        'device': device,
    }
    run_tgda_attack(**tgda_config)
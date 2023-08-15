import torch
from .eval import *
from .attacks import *
from .datasets import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    torch.cuda.empty_cache()

    # train_cifar_tgda_config = {
    #     'poisoner_model': CIFAR10Poisoner(),
    #     'poisoner_load_file': 'poisoner_model_epoch_1299.tar',
    #     'poisoner_model_steps': 1,
    #     'poisoner_model_step_size': 0.1,
    #     'poisoned_model': CIFAR10PoisonedResnetWithPretraining(),
    #     'poisoned_model_steps': 10,
    #     'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9),
    #     'train_loss': ce_loss,
    #     'test_loss': poisoned_ce_loss,
    #     'eval_metric': accuracy,
    #     'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
    #     'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
    #     'train_ratio':0.7,
    #     'epsilon': 0.03,
    #     'epochs': 2000,
    #     'print_epochs': 1,
    #     'save_epochs': 20,
    #     'save_folder': './checkpoints/tgda/cifar10/resnet_pretrained/0.1',
    #     'device': device,
    # }

    # test_cifar_tgda_config = {
    #     'poisoner_model': CIFAR10Poisoner(),
    #     'poisoner_load_epoch': 199,
    #     'poisoned_model': CIFAR10PoisonedResnetWithPretraining(),
    #     'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9),
    #     'train_loss': ce_loss,
    #     'eval_metric': accuracy,
    #     'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
    #     'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
    #     'epsilon': 0.03,
    #     'epochs': 2000,
    #     'print_epochs': 1,
    #     'save_epochs': 20,
    #     'save_folder': './checkpoints/tgda/cifar10/resnet_pretrained/0.1',
    #     'device': device,
    # }

    # train_mnist_tgda_config = {
    #     'poisoner_model': MNISTPoisoner(),
    #     'poisoner_load_file': None,
    #     'poisoner_model_steps': 1,
    #     'poisoner_model_step_size': 0.1,
    #     'poisoned_model': MNISTPoisonedLR(),
    #     'poisoned_model_steps': 10,
    #     'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
    #     'train_loss': ce_loss,
    #     'test_loss': poisoned_ce_loss,
    #     'eval_metric': accuracy,
    #     'train_loader': get_mnist_train_loader('./data', batch_size=1000),
    #     'test_loader': get_mnist_test_loader('./data', batch_size=1000),
    #     'train_ratio':0.7,
    #     'epsilon': 0.03,
    #     'epochs': 200,
    #     'print_epochs': 1,
    #     'save_epochs': 20,
    #     'save_folder': './checkpoints/tgda/mnist/lr/',
    #     'device': device,
    # }


    # test_mnist_tgda_config = {
    #     'poisoner_model': MNISTPoisoner(),
    #     'poisoner_load_epoch': 199,
    #     'poisoned_model': MNISTPoisonedLR(),
    #     'poisoned_model_optimizer': lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
    #     'train_loss': ce_loss,
    #     'eval_metric': accuracy,
    #     'train_loader': get_mnist_train_loader('./data', batch_size=1000),
    #     'test_loader': get_mnist_test_loader('./data', batch_size=1000),
    #     'epsilon': 0.03,
    #     'epochs': 200,
    #     'print_epochs': 1,
    #     'save_epochs': 20,
    #     'save_folder': './checkpoints/tgda/mnist/lr/',
    #     'device': device,
    # }

    # attack_tgda(**train_cifar_tgda_config)
    # test_tgda(**test_cifar_tgda_config)


    # attack_tgda(**train_mnist_tgda_config)
    # test_tgda(**test_mnist_tgda_config)


    attack_pc_config = {
        'model': CIFAR10PoisonedResnetWithPretraining(),
        'train_loss': ce_loss,
        'optimizer': lambda params: torch.optim.SGD(params, lr=0.001, momentum=0.9),
        'pc_optimizer': lambda params: ParamCorrupter(params, lr=0.1, eps=1, LP='l2'),
        'eval_metric': accuracy,
        'train_loader': get_cifar10_train_loader('./data', batch_size=1000),
        'test_loader': get_cifar10_test_loader('./data', batch_size=1000),
        'epochs': 20,
        'print_epochs': 2,
        'save_folder': './checkpoints/gc/cifar10/pc/',
        'device': device,
    }

    attack_pc(**attack_pc_config)

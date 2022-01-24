
import sys
import os
sys.path.insert(0,'../../utils')
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import wandb
from models import Linear_regime_1, Linear_regime_2, Linear_regime_3
from models import Conv_regime_1, Conv_regime_3
from models import VGG_regime_1, VGG_regime_3
from models import Resnet_regime_1, Resnet_regime_3
from model_vgg import VGG_classic
from model_resnet import Resnet20_classic
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from data import BostonDataset
import pickle as pkl
import os
import torch
import numpy as np

def load_boston(batch_size, num_works):
    dataset = BostonDataset()
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    return trainloader, trainloader

def load_mnist(batch_size, num_works):
    dataset = MNIST('../', download=True, transform=transforms.ToTensor(), train=True)
    trainset = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    dataset = MNIST('../', download=True, transform=transforms.ToTensor(), train=False)
    testset = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    return trainset, testset

def load_cifar(batch_size, num_works):
    trainset = CIFAR10('../', download=False, transform=transforms.ToTensor(), train=True)
    trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    testset = CIFAR10('../', download=False, transform=transforms.ToTensor(), train=False)
    testset = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainset, testset

def limit_size(dataset, size):
    indexes = np.random.choice(range(0, dataset.dataset.__len__()), size=size, replace=False)
    dataset.dataset.targets = dataset.dataset.targets[indexes]
    dataset.dataset.data = dataset.dataset.data[indexes] 
    return dataset  

def load_data(batch_size, dataset_name, num_works, train_params, model_params):
    if dataset_name == 'MNIST':
        trainset, testset = load_mnist(batch_size, num_works)
        model_params['hin'] = 28
        model_params['in_size'] = 1
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'CIFAR10':
        trainset, testset = load_cifar(batch_size, num_works)
        model_params['hin'] = 32
        model_params['in_size'] = 3
        model_params['out_size'] = 10
        train_params['save_acc'] = True
    elif dataset_name == 'BOSTON':
        trainset, testset = load_boston(batch_size, num_works)
        model_params['hin'] = 1
        model_params['in_size'] = 13
        model_params['out_size'] = 1
        train_params['save_acc'] = False
    else:
        raise ValueError('To implement')
    if train_params['limit_p'] != None:
        trainset = limit_size(trainset, train_params['limit_p'])

    train_params['nb_batches'] = trainset.__len__()
    train_params['p'] = trainset.dataset.__len__()
    return trainset, testset

def get_model(model_name, dist_params, train_params, model_params):
    if model_name == 'Linear_regime_1':
        model_params['in_size'] = model_params['in_size'] * model_params['hin']**2
        return Linear_regime_1(dist_params, train_params, model_params)

    elif model_name == 'Linear_regime_2':
        model_params['in_size'] = model_params['in_size'] * model_params['hin']**2
        return Linear_regime_2(dist_params, train_params, model_params)

    elif model_name == 'Linear_regime_3':
        model_params['in_size'] = model_params['in_size'] * model_params['hin']**2
        return Linear_regime_3(dist_params, train_params, model_params)

    elif model_name == 'Conv_regime_1':
        return Conv_regime_1(dist_params, train_params, model_params)

    elif model_name == 'Conv_regime_3':
        return Conv_regime_3(dist_params, train_params, model_params)

    elif model_name == 'VGG_regime_1':
        return VGG_regime_1(dist_params, train_params, model_params)

    elif model_name == 'VGG_regime_3':
        return VGG_regime_3(dist_params, train_params, model_params)
    elif model_name == 'Resnet_regime_1':
        return Resnet_regime_1(dist_params, train_params, model_params)
    elif model_name == 'Resnet_regime_3':
        return Resnet_regime_3(dist_params, train_params, model_params)
    elif model_name == 'VGG_classic':
        return VGG_classic(train_params, model_params)
    elif model_name == 'Resnet20_classic':
        return Resnet20_classic(train_params, model_params)
    else:
        raise ValueError(f'To implement: {model_name}')

def get_exp_name(train_params, model_params):
    w = model_params['w']
    alpha = train_params['alpha']
    return f'w_{w}_alpha_{alpha}'

def get_trainer(nb_epochs, wandb_logger, lr_monitor, exp_name):
    if torch.cuda.is_available():
        trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger, callbacks=[lr_monitor], weights_save_path = exp_name)
    else:
        trainer = pl.Trainer(max_epochs=nb_epochs, logger= wandb_logger)
    return trainer

def save_weights(mu, std, exp_name):
    pkl.dump({'mu': mu, 'std': std}, open(exp_name, 'wb'))

def main(project_name, model_name, dataset_name, num_works, batch_size, dist_params, train_params, model_params):
    trainset, testset = load_data(batch_size, dataset_name, num_works, train_params, model_params)
    model = get_model(model_name, dist_params, train_params, model_params)
    exp_name = get_exp_name(train_params, model_params)
    wandb_logger = WandbLogger(name=exp_name,project=project_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = get_trainer(train_params['nb_epochs'], wandb_logger, lr_monitor, exp_name)
    trainer.fit(model, trainset, testset)
    result = trainer.test(model, testset)
    wandb.finish()
    save_weights(model.mu, model.std, exp_name)
    return model





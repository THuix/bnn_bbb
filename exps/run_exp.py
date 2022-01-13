import os
os.chdir("..")
import numpy as np
from numpy.lib.function_base import average
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader, random_split
import os
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from utils.models import Model_regime_1, Model_regime_2, Model_regime_3, NN
import numpy as np
import argparse

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
out_size = 10
init_mu_post = 0.
sigma_prior = 1.
init_rho_post = np.log(np.exp(sigma_prior)-1)
mu_prior = 0.
batch_size = 1024
nb_epochs = 200
num_works=8

def load_mnist(batch_size):
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    trainset = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
    testset = DataLoader(dataset, batch_size=batch_size, num_workers=num_works)
    return trainset, testset

def load_cifar(batch_size):
    trainset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    trainset = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    
    testset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
    testset = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainset, testset    

# Functions & Classes
def load_data(batch_size, alpha, regime, nb_samples, lr, N, dataset_name):
    if dataset_name == 'MNIST':
        trainset, testset = load_mnist(batch_size)
        in_size = 28*28
    elif dataset_name == 'CIFAR10':
        trainset, testset = load_cifar(batch_size)
        in_size = 32*32*3
    else:
        raise ValueError('To implement')
    p = trainset.dataset.__len__()
    nb_batches = len(trainset)

    if regime == 1:
        alpha = N / p
        
    dist_params = {'init_rho_post': init_rho_post, 'init_mu_post': init_mu_post, 'sigma_prior': sigma_prior, 'mu_prior':mu_prior}
    train_params = {'lr': lr, 'nb_samples': nb_samples, 'nb_batches': nb_batches, 'criterion': criterion, "alpha": alpha}
    return trainset, testset, p, dist_params, train_params, train_params['alpha'], lr, in_size

def get_model(regime, p, dist_params, train_params, lr, N, in_size):
    if regime == 1:
        bnn = Model_regime_1(in_size, out_size, N, p, dist_params, train_params)
    elif regime == 2:
        bnn = Model_regime_2(in_size, out_size, N, p, dist_params, train_params)
    elif regime == 3:
        bnn = Model_regime_3(in_size, out_size, N, p, dist_params, train_params)
    elif regime == 'nn':
        bnn = NN(in_size, out_size, N, criterion, lr)
    else:
        raise ValueError('To implement')
    return bnn

def get_exp_name(regime, N, p, alpha, lr, nb_samples):
    if regime == 1:
        return f"regime_{regime}_N_{N}_p_{p}_sigmaprior_{sigma_prior}_lr_{lr}_nb_samples_{nb_samples}"
    elif regime == 2:
        return f"regime_{regime}_N_{N}_p_{p}_alpha_{alpha}_sigmaprior_{sigma_prior}_lr_{lr}_nb_samples_{nb_samples}"
    elif regime == 3:
        return f"regime_{regime}_N_{N}_p_{p}_alpha_{alpha}_sigmaprior_{sigma_prior}_lr_{lr}_nb_samples_{nb_samples}"
    else:
        raise ValueError('To implement')

def save_config_file(N, p, alpha, nb_samples, lr):
    wandb.config.N = N
    wandb.config.p = p 
    wandb.config.alpha = alpha
    wandb.config.nb_samples = nb_samples
    wandb.config.lr = lr
    wandb.config.batch_size = batch_size
    wandb.config.sigma_prior = sigma_prior 
    wandb.config.init_rho_post = init_rho_post
    wandb.config.init_mu_post = init_mu_post
    wandb.config.mu_prior = mu_prior
    wandb.finish()

def main(N, lr, nb_samples, alpha, regime, project_name, dataset_name):
    trainset, testset, p, dist_params, train_params, alpha, lr, in_size = load_data(batch_size, alpha, regime, nb_samples, lr, N, dataset_name)
    model = get_model(regime, p, dist_params, train_params, lr, N, in_size)
    exp_name = get_exp_name(regime, N, p, alpha, lr, nb_samples)
    wandb_logger = WandbLogger(name=exp_name,project=project_name)
    trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger)
    #trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger, strategy="ddp")
    #trainer = pl.Trainer(max_epochs=nb_epochs, logger= wandb_logger, track_grad_norm=2)
    trainer.fit(model, trainset, testset)
    result = trainer.test(model, testset)
    save_config_file(N, p, alpha, nb_samples, lr)

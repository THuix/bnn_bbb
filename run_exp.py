import numpy as np
from numpy.lib.function_base import average
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import os
from pytorch_lightning.loggers.wandb import WandbLogger
import wandb
from models import Model_regime_1, Model_regime_2, Model_regime_3, NN
import numpy as np

# Hyperparameters
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
in_size = 28*28
out_size = 10
init_mu_post = 0.
sigma_prior = 1.
init_rho_post = np.log(np.exp(sigma_prior)-1)
mu_prior = 0.
batch_size = 1024
nb_epochs = 200

# Functions & Classes
def load_data(batch_size, alpha, regime, nb_samples, lr, N):
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
    trainset = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
    testset = DataLoader(dataset, batch_size=batch_size, num_workers=8)
    p = trainset.dataset.__len__()
    nb_batches = len(trainset)

    if regime == 1:
        alpha = N / p
        
    dist_params = {'init_rho_post': init_rho_post, 'init_mu_post': init_mu_post, 'sigma_prior': sigma_prior, 'mu_prior':mu_prior}
    train_params = {'lr': lr, 'nb_samples': nb_samples, 'nb_batches': nb_batches, 'criterion': criterion, "alpha": alpha}
    return trainset, testset, p, nb_batches, dist_params, train_params, train_params['alpha'], lr

def get_model(regime, p, dist_params, train_params, lr, N):
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

def main(N, lr, nb_samples, alpha, regime):
    trainset, testset, p, nb_batches, dist_params, train_params, alpha, lr = load_data(batch_size, alpha, regime, nb_samples, lr, N)
    model = get_model(regime, p, dist_params, train_params, lr, N)
    exp_name = get_exp_name(regime, N, p, alpha, lr, nb_samples)
    wandb_logger = WandbLogger(name=exp_name,project='BNN_regimes')
    trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger)
    #trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger, strategy="ddp")
    #trainer = pl.Trainer(max_epochs=nb_epochs, logger= wandb_logger, track_grad_norm=2)
    trainer.fit(model, trainset, testset)

    result = trainer.test(model, testset)
    wandb.finish()

def exp_1(range_N):
    lr = 1e-1
    nb_samples = 10
    alpha = None
    regime = 1
    for N in range_N:
        main(N, lr, nb_samples, alpha, regime)

def exp_3(range_N, alpha):
    lr = 1e-1
    nb_samples = 1
    regime = 3
    for N in range_N:
        main(N, lr, nb_samples, alpha, regime)


if __name__ == '__main__':
    exp_1([1000, 5000, 10000])


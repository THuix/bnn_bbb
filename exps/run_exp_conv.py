import sys
import os
sys.path.insert(0,'../utils')
import numpy as np
from run_exp import load_mnist, load_cifar, load_boston
from models import Conv_Model_regime_1, Conv_Model_regime_3
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import wandb

out_size = 10
padding = 0
dilation = 1
kernel_size = 3
stride = 1
init_mu_post = 0.
sigma_prior = 1.
mu_prior = 0.
batch_size = 128

def get_model(regime, hidden_channels, p, dist_params, train_params, conv_params, in_size):
    if regime == 1:
        bnn = Conv_Model_regime_1(in_size, out_size, hidden_channels, p, dist_params, train_params, conv_params)
    elif regime == 3:
        bnn = Conv_Model_regime_3(in_size, out_size, hidden_channels, p, dist_params, train_params, conv_params)
    else:
        raise ValueError('To implement')
    return bnn

# Functions & Classes
def load_data(batch_size, dataset_name):
    if dataset_name == 'MNIST':
        trainset, testset = load_mnist(batch_size)
        in_size = 1
        hin = 28
    elif dataset_name == 'CIFAR10':
        trainset, testset = load_cifar(batch_size)
        in_size = 3
        hin = 32
    else:
        raise ValueError('To implement')

    return trainset, testset, in_size, hin


def get_nb_neurons(out_channels, hin, p, d, k, s):
    hout = int((hin + 2*p - d * (k-1) - 1 ) / s + 1)
    return out_channels * hout * hout

def init_params(trainset, alpha, regime, nb_samples, lr, hidden_channels, criterion, init_rho_post, hin):

    p = trainset.dataset.__len__()
    nb_batches = len(trainset)

    N = get_nb_neurons(hidden_channels, hin, padding, dilation, kernel_size, stride)

    if regime == 1:
        alpha = N / p
        
    dist_params = {'init_rho_post': init_rho_post, 'init_mu_post': init_mu_post, 'sigma_prior': sigma_prior, 'mu_prior':mu_prior}
    train_params = {'lr': lr, 'nb_samples': nb_samples, 'nb_batches': nb_batches, 'criterion': criterion, "alpha": alpha}
    conv_params = {'padding' : padding, 'dilation': dilation, 'stride': stride, 'hin': hin, 'N': N, 'kernel_size': kernel_size}
    return p, dist_params, train_params, conv_params, train_params['alpha'], lr

def get_exp_name(regime, N, p, alpha, lr, nb_samples):
    if regime == 1:
        return f"regime_{regime}_N_{N}_p_{p}_sigmaprior_{sigma_prior}_lr_{lr}_nb_samples_{nb_samples}"
    elif regime == 3:
        return f"regime_{regime}_N_{N}_p_{p}_alpha_{alpha}_sigmaprior_{sigma_prior}_lr_{lr}_nb_samples_{nb_samples}"
    else:
        raise ValueError('To implement')

def save_config_file(hidden_channels, N, p, alpha, nb_samples, lr, model, init_rho_post):
    wandb.config.hidden_channels = hidden_channels
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

def main_conv(hidden_channels, lr, nb_samples, alpha, regime, project_name, dataset_name, criterion, nb_epochs, init_rho_post):
    trainset, testset, in_size, hin = load_data(batch_size, dataset_name)
    p, dist_params, train_params, conv_params, train_params['alpha'], lr = init_params(trainset, alpha, regime, nb_samples, lr, hidden_channels, criterion, init_rho_post, hin)
    model = get_model(regime, hidden_channels, p, dist_params, train_params, conv_params, in_size)
    exp_name = get_exp_name(regime, conv_params['N'], p, alpha, lr, nb_samples)
    wandb_logger = WandbLogger(name=exp_name,project=project_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #trainer = pl.Trainer(max_epochs=nb_epochs, logger= wandb_logger, track_grad_norm=2)
    trainer = pl.Trainer(gpus=-1, max_epochs=nb_epochs, logger= wandb_logger, callbacks=[lr_monitor], weights_save_path = exp_name)
    trainer.fit(model, trainset, testset)
    result = trainer.test(model, testset)
    save_config_file(hidden_channels, conv_params['N'], p, alpha, nb_samples, lr, model, init_rho_post)


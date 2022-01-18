from numpy.lib import stride_tricks
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
from utils_models import BNN
from layers import Conv_bnn, Linear_bnn

class Conv_BNN(BNN):
    def __init__(self, regime, dist_params, train_params, model_params):
        
        self.dist_params = self.check_params(dist_params, ['init_rho_post', 'init_mu_post', 'sigma_prior', 'mu_prior'])
        self.train_params = self.check_params(train_params, ['lr', 'nb_samples', 'nb_batches', 'criterion', 'alpha', 'p'])
        self.model_params = self.check_params(model_params, ['N_last_layer', 'in_size', 'out_size', 'hin', 'padding', 'stride', 'dilation', 'kernel_size'])

        super(Conv_BNN, self).__init__(dist_params, train_params, model_params, regime)
        
        self.save_hyperparameters()

        self.seq = nn.Sequential(
            Conv_bnn(self.model_params['in_size'],
                     self.model_params['N_last_layer'],
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       stride = self.conv_params['stride'],
                       padding = self.conv_params['padding'],
                       dilation = self.conv_params['dilation'],
                       kernel_size = self.conv_params['kernel_size'],
                       init_type='normal',
                       regime=regime),
            nn.ReLU(),
            nn.Flatten(),
            Linear_bnn(self.model_params['N_last_layer'],
                       self.model_params['out_size'],
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       init_type='normal',
                       regime=regime,
                       bias = False))

        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        self.regime = regime
        self.save_hist = False
        self.do_flatten = False
        self.T = self.get_temperature(regime)
        self.save_hyperparameters()  
    

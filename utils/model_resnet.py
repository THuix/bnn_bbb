from utils_models import BNN
from layers import Conv_bnn, Linear_bnn
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from model_linear import NN

class Resnet_bloc(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, conv_in_identity, dist_params, model_params, regime):
        super(Resnet_bloc, self).__init__()
        self.seq = nn.Sequential(
            Conv_bnn(in_channels,
                     out_channels,
                     dist_params['init_rho_post'],
                     dist_params['init_mu_post'],
                     dist_params['sigma_prior'],
                     dist_params['mu_prior'],
                     stride = stride,
                     padding = 1,
                     dilation = 1,
                     kernel_size = ks,
                     init_type='normal',
                     regime=regime),
            nn.ReLU(),
            Conv_bnn(out_channels,
                     out_channels,
                     dist_params['init_rho_post'],
                     dist_params['init_mu_post'],
                     dist_params['sigma_prior'],
                     dist_params['mu_prior'],
                     stride = 1,
                     padding = 1,
                     dilation = 1,
                     kernel_size = ks,
                     init_type='normal',
                     regime=regime))
        if conv_in_identity:
            self.seq_identity = nn.Sequential(
                Conv_bnn(in_channels,
                         out_channels,
                         dist_params['init_rho_post'],
                         dist_params['init_mu_post'],
                         dist_params['sigma_prior'],
                         dist_params['mu_prior'],
                         stride = 2,
                         padding = 0,
                         dilation = 1,
                         kernel_size = 1,
                         init_type='normal',
                         regime=regime))
        else:
            self.seq_identity = nn.Sequential()

    def forward(self, x):
        x_conv = self.seq(x)
        x_identity = self.seq_identity(x)
        x_out = x_conv + x_identity
        return F.relu(x_out)


def create_resnet_seq(dist_params, model_params, regime):
    return nn.Sequential(
        Conv_bnn(model_params['in_size'],
                16,
                dist_params['init_rho_post'],
                dist_params['init_mu_post'],
                dist_params['sigma_prior'],
                dist_params['mu_prior'],
                stride = 1,
                padding = 1,
                dilation = 1,
                kernel_size = 3,
                init_type='normal',
                regime=regime),
        nn.ReLU(),
        Resnet_bloc(16, 16, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(16, 16, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(16, 16, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(16, 32, 3, 2, True, dist_params, model_params, regime),
        Resnet_bloc(32, 32, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(32, 32, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(32, 64, 3, 2, True, dist_params, model_params, regime),
        Resnet_bloc(64, 64, 3, 1, False, dist_params, model_params, regime),
        Resnet_bloc(64, 64, 3, 1, False, dist_params, model_params, regime),
        nn.AvgPool2d(8),
        nn.Flatten(),
        Linear_bnn(64,
                10,
                dist_params['init_rho_post'],
                dist_params['init_mu_post'],
                dist_params['sigma_prior'],
                dist_params['mu_prior'],
                init_type='normal',
                regime=regime,
                bias = True))
                       

class Resnet20(BNN):
    def __init__(self, regime, dist_params, train_params, model_params):
        
        self.dist_params = self.check_params(dist_params, ['init_rho_post', 'init_mu_post', 'sigma_prior', 'mu_prior'])
        self.train_params = self.check_params(train_params, ['lr', 'nb_samples', 'nb_batches', 'criterion', 'alpha', 'p'])
        self.model_params = self.check_params(model_params, ['in_size', 'out_size', 'hin'])

        super(Resnet20, self).__init__(dist_params, train_params, model_params, regime)


        self.model_params['N_last_layer'] = 64 

        self.seq = nn.Sequential(*create_resnet_seq(dist_params, model_params, regime))
        
        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        
        if regime == 1:
            self.train_params['alpha'] = self.model_params['w'] / self.train_params['p']

        self.regime = regime
        self.save_hist = False
        self.do_flatten = False
        self.T = self.get_temperature(regime)
        self.save_hyperparameters() 



class Resnet_bloc_classic(nn.Module):
    def __init__(self, in_channels, out_channels, ks, stride, conv_in_identity):
        super(Resnet_bloc, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ks, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, ks, stride=1, padding=1))
        if conv_in_identity:
            self.seq_identity = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1, stride=2))
        else:
            self.seq_identity = nn.Sequential()

    def forward(self, x):
        x_conv = self.seq(x)
        x_identity = self.seq_identity(x)
        x_out = x_conv + x_identity
        return F.relu(x_out)


def create_resnet_classic_seq(model_params):
    return nn.Sequential(
        nn.Conv2d(model_params['in_size'], 16, 3, padding=1),
        nn.ReLU(),
        Resnet_bloc_classic(16, 16, 3, 1, False),
        Resnet_bloc_classic(16, 16, 3, 1, False),
        Resnet_bloc_classic(16, 16, 3, 1, False),
        Resnet_bloc_classic(16, 32, 3, 2, True),
        Resnet_bloc_classic(32, 32, 3, 1, False),
        Resnet_bloc_classic(32, 32, 3, 1, False),
        Resnet_bloc_classic(32, 64, 3, 2, True),
        Resnet_bloc_classic(64, 64, 3, 1, False),
        Resnet_bloc_classic(64, 64, 3, 1, False),
        nn.AvgPool2d(8),
        nn.Flatten(),
        nn.Linear(64, 10))
                       

class Resnet20_classic(NN):
    def __init__(self, train_params, model_params):
        self.train_params = train_params
        self.model_params = model_params
        super(Resnet20_classic, self).__init__(64,
                                               self.train_params['criterion'],
                                               self.train_params['lr'])


        self.model_params['N_last_layer'] = 64 
        self.seq = nn.Sequential(*create_resnet_classic_seq(model_params))
        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        self.save_hist = False
        self.do_flatten = False
        self.save_hyperparameters() 

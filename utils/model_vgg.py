from numpy.lib import stride_tricks
from torch._C import INSERT_FOLD_PREPACK_OPS
import torchmetrics
from torch import nn
import pytorch_lightning as pl
import torch
import numpy as np
from utils_models import BNN
from model_linear import NN
from layers import Conv_bnn, Linear_bnn


all_layers = {
 11: [('C', True, 0, 64, 3), ('M'), ('C', False, 64, 128, 3), ('M'), ('C', False, 128, 256, 3), ('C', False, 256, 256, 3),
        ('M'), ('C', False, 256, 512, 3), ('C', False, 512, 512, 3), ('M'), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
        ('M'), ('F'), ('L', True, 0, 512, True), ('L', False, 512, 512, True), ('L', False, 512, 10, False)],

 13: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), ('M'), ('C', False, 64, 128, 3), ('C', False, 128, 128, 3), ('M'),
        ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), ('M'), ('C', False, 256, 512, 3), ('C', False, 512, 512, 3),
        ('M'), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('M'), ('F'),('L', True, 0, 512, True),
        ('L', False, 512, 512, True), ('L', False, 512, 10, False)],

 16: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), ('M'), ('C', False, 64, 128, 3), ('C', False, 128,  128, 3), ('M'),
        ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), ('C', False, 256, 256, 3), ('M'), ('C', False, 256, 512, 3),
        ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('M'), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
        ('C', False, 512, 512, 3), ('M'), ('F'),('L', True, 0, 512, True), ('L', False, 512, 512, True),
        ('L', False, 512, 10, False)],

 19: [('C', True, 0, 64, 3), ('C', False, 64, 64, 3), ('M'), ('C', False, 64, 128, 3), ('C', False, 128, 128, 3), 
        ('M'), ('C', False, 128, 256, 3), ('C', False, 256, 256, 3), ('C', False, 256, 256, 3), ('C', False, 256, 256, 3),
        ('M'), ('C', False, 256, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
        ('M'), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3), ('C', False, 512, 512, 3),
        ('M'), ('F'), ('L', True, 0, 512, True), ('L', False, 512, 512, True), ('L', False, 512, 10, False)]
 }

class VGG(BNN):
    def __init__(self, regime, dist_params, train_params, model_params):
        
        self.dist_params = self.check_params(dist_params, ['init_rho_post', 'init_mu_post', 'sigma_prior', 'mu_prior'])
        self.train_params = self.check_params(train_params, ['lr', 'nb_samples', 'nb_batches', 'criterion', 'alpha', 'p'])
        self.model_params = self.check_params(model_params, ['VGG_type', 'in_size', 'out_size', 'hin'])

        super(VGG, self).__init__(dist_params, train_params, model_params, regime)


        self.model_params['N_last_layer'] = 512 

        self.seq = nn.Sequential(*self.create_seq(model_params['VGG_type'], dist_params, regime, model_params['in_size'], model_params['hin']))
        
        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        
        if regime == 1:
            self.train_params['alpha'] = self.model_params['w'] / self.train_params['p']
        self.regime = regime
        self.save_hist = False
        self.do_flatten = False
        self.T = self.get_temperature(regime)
        self.save_hyperparameters()  


    def create_seq(self, vgg_type, dist_params, regime, in_size, hin):

        layers = all_layers[vgg_type]
        seq = []
        for layer in layers:
            if layer[0] == 'C': # conv layer
                seq.append(nn.Dropout(0.2))
                seq.append(
                    Conv_bnn(   in_size if layer[1] else layer[2],
                                layer[3],
                                dist_params['init_rho_post'],
                                dist_params['init_mu_post'],
                                dist_params['sigma_prior'],
                                dist_params['mu_prior'],
                                stride = self.model_params['stride'],
                                padding = self.model_params['padding'],
                                dilation = self.model_params['dilation'],
                                kernel_size = layer[4],
                                init_type='normal',
                                regime=regime,
                                bias=False))
                seq.append(nn.ReLU(inplace=True))

            elif layer[0] == 'M': #maxpooling
                seq.append(nn.MaxPool2d(2, stride=2))

            elif layer[0] == 'L': #Linear
                seq.append(nn.Dropout(0.5))
                seq.append(
                    Linear_bnn(512 if layer[1] else layer[2],
                    layer[3],
                    dist_params['init_rho_post'],
                    dist_params['init_mu_post'],
                    dist_params['sigma_prior'],
                    dist_params['mu_prior'],
                    init_type='normal',
                    regime=regime,
                    bias = True)
                    )
                if layer[4]:
                    seq.append(nn.ReLU(inplace=True))

            elif layer[0] == 'F': # flatten
                seq.append(nn.Flatten())
            else:
                raise ValueError(f'layer name not find: {layer[0]}')
        return seq



class VGG_classic(NN):
    def __init__(self, train_params, model_params):
        self.train_params = train_params
        self.model_params = model_params
        super(VGG_classic, self).__init__(512,
                                          self.train_params['criterion'],
                                          self.train_params['lr'])


        self.seq = nn.Sequential(*self.create_seq(model_params['VGG_type'], model_params['in_size']))
        
        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        self.save_hist = False
        self.do_flatten = False
        self.save_hyperparameters()  

    def create_seq(self, vgg_type, in_size):

        layers = all_layers[vgg_type]
        seq = []
        for layer in layers:
            if layer[0] == 'C': # conv layer
                seq.append(nn.Dropout(0.2))
                seq.append(
                    nn.Conv2d(in_size if layer[1] else layer[2],
                              layer[3],
                              stride = self.model_params['stride'],
                              padding = self.model_params['padding'],
                              dilation = self.model_params['dilation'],
                              kernel_size = layer[4],
                              bias=False))
                seq.append(nn.ReLU(inplace=True))

            elif layer[0] == 'M': #maxpooling
                seq.append(nn.MaxPool2d(2, stride=2))

            elif layer[0] == 'L': #Linear
                seq.append(nn.Dropout(0.5))
                seq.append(
                    nn.Linear(512 if layer[1] else layer[2],
                    layer[3],
                    bias=True)
                    )
                if layer[4]:
                    seq.append(nn.ReLU(inplace=True))

            elif layer[0] == 'F': # flatten
                seq.append(nn.Flatten())
            else:
                raise ValueError(f'layer name not find: {layer[0]}')
        return seq



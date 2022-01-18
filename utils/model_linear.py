
import torchmetrics
from torch import nn
import numpy as np
import pytorch_lightning as pl
import torch
from utils_models import BNN
from layers import Linear_bnn

class Linear_BNN(BNN):
    def __init__(self, regime, dist_params, train_params, model_params):

        self.dist_params = self.check_params(dist_params, ['init_rho_post', 'init_mu_post', 'sigma_prior', 'mu_prior'])
        self.train_params = self.check_params(train_params, ['lr', 'nb_samples', 'nb_batches', 'criterion', 'alpha', 'p'])
        self.model_params = self.check_params(model_params, ['in_size', 'out_size', 'N_last_layer'])

        super(Linear_BNN, self).__init__(dist_params, train_params, model_params, regime)
               
        self.seq = nn.Sequential(
            Linear_bnn(model_params['in_size'],
                       model_params['N_last_layer'],
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       init_type='normal',
                       regime=regime,
                       bias = False),
            nn.ReLU(),
            Linear_bnn(model_params['N_last_layer'],
                       model_params['out_size'],
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       init_type='normal',
                       regime=regime,
                       bias = False))

        self.model_params['w'] = np.sum([m.flatten().detach().cpu().numpy().shape for m in self.parameters()])
        if regime == 1:
            self.train_params['alpha'] = self.model_params['w'] / self.train_params['p']
        self.regime = regime
        self.save_hist = True
        self.do_flatten = True
        self.T = self.get_temperature(regime)
        self.save_hyperparameters()  

class NN(pl.LightningModule):
    def __init__(self, in_size, out_size, N, criterion, lr):
        super(NN, self).__init__()
        self.save_hyperparameters()
          
        self.seq = nn.Sequential(
            nn.Linear(in_size, N, bias=False),
            nn.ReLU(),
            nn.Linear(N, out_size, bias=False))

        self.accuracy = torchmetrics.Accuracy()
        self.criterion = criterion
        self.lr = lr
        self.N = N

    def forward(self, x):
        pass

    def step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size()[0], -1)
        pred = self.seq(x) / self.N

        loss = self.criterion(pred, y)
        self.accuracy.update(pred, y)
        logs = {
            'acc': self.accuracy.compute(),
            'nll': loss.item(),
        }
        
        return loss, logs       
    
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.lr)
        return optimizer



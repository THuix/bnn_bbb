import torchmetrics
from torch import nn
import numpy as np
from layers import Linear_bnn
from security import check, check_is_tensor
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class BNN(pl.LightningModule):
    def __init__(self, in_size, out_size, N, p, dist_params, train_params, regime):
        super(BNN, self).__init__()
        self.save_hyperparameters()
        
        self.dist_params = self.check_params(dist_params, ['init_rho_post', 'init_mu_post', 'sigma_prior', 'mu_prior'])
        self.train_params = self.check_params(train_params, ['lr', 'nb_samples', 'nb_batches', 'criterion', 'alpha'])

        self.seq = nn.Sequential(
            Linear_bnn(in_size,
                       N,
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       N,
                       p,
                       self.train_params['alpha'],
                       init_type='fixed',
                       regime=regime,
                       bias = False),
            nn.ReLU(),
            Linear_bnn(N,
                       out_size,
                       self.dist_params['init_rho_post'],
                       self.dist_params['init_mu_post'],
                       self.dist_params['sigma_prior'],
                       self.dist_params['mu_prior'],
                       N,
                       p,
                       self.train_params['alpha'],
                       init_type='normal',
                       regime=regime,
                       bias = False))
        self.accuracy = torchmetrics.Accuracy()
        self.ECE = torchmetrics.CalibrationError(n_bins=15, norm='l1')
        self.N = N
        self.regime = regime
        self.p = p
        self.out_size = out_size
        self.T = self.get_temperature(regime)

    def get_temperature(self, regime):
        if regime == 1 or regime == 2:
            return 1
        elif regime == 3:
            return self.train_params['alpha'] * self.p / self.N
        else:
            raise ValueError('To implement')

    def check_params(self, params, true_params):
        for p in true_params:
            if not(p in params):
                raise ValueError(f"{p} is missing in {params}")
        return params

    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        predictions = torch.tensor([self.seq(x) for _ in range(self.train_params['nb_samples'])])
        raise ValueError(predictions.mean(dim=0).size())
        return predictions.mean(dim=0) / self.N
    
    def _get_kl(self):
        kl = 0
        for module in self.modules():
            if hasattr(module, 'appro_kl'):
                kl += module.appro_kl
        check_is_tensor(kl, 'KL')
        check(kl)
        return kl
    
    def _step_1_sample(self, x, y):
        pred = self.seq(x) / self.N**2
        nll = self.train_params['criterion'](pred, y) / self.T
        kl = self._get_kl() / self.train_params['nb_batches']
        obj_loss = nll + kl
        return obj_loss, nll, kl, pred

    def re_balance_loss(self, loss):
        if self.regime == 1 or self.regime == 3: 
            return self.train_params['alpha'] * loss * self.train_params['nb_batches'] / self.N
        elif self.regime == 2:
            return loss * self.train_params['nb_batches'] / self.p
        else:
            raise ValueError('To implement')

    def step(self, batch, batch_idx):
        x, y = batch
        x = x.reshape(x.size()[0], -1)

        obj_loss = torch.zeros(1, requires_grad=True).type_as(x)
        nll = torch.zeros(1).type_as(x)
        kl = torch.zeros(1).type_as(x)
        pred = torch.zeros((x.size()[0], self.out_size)).type_as(x)

        for idx in range(self.train_params['nb_samples']):
            o, n, k, p = self._step_1_sample(x, y)
            obj_loss = obj_loss + o / self.train_params['nb_samples']
            nll = nll + n / self.train_params['nb_samples']
            kl = kl + k / self.train_params['nb_samples']
            pred = pred + p / self.train_params['nb_samples']
        
        obj_loss = self.re_balance_loss(obj_loss)
        nll = self.re_balance_loss(nll)
        kl = self.re_balance_loss(kl)

        self.accuracy.update(pred, y)
        self.ECE.update(pred, y)
        logs = {
            'acc': self.accuracy.compute(),
            'ece': self.ECE.compute(),
            "obj": obj_loss,
            "kl": kl,
            "nll": nll,
            "ratio_nll_kl": nll / kl,
        }
        
        return obj_loss, logs   
    
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def plot_hist(self, values, title, name):
        fig = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.hist(values, bins='auto')
        ax.set_title(title)
        canvas.draw()
        buf = canvas.buffer_rgba()
        X = np.asarray(buf)
        wandb.log({name: wandb.Image(X)})

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def test_epoch_end(self, output):
        mu_1 = self.seq[0].weight_mu.flatten().detach().cpu().numpy()
        mu_2 = self.seq[2].weight_mu.flatten().detach().cpu().numpy()
        std_1 = self.seq[0].rho_to_std(self.seq[0].weight_rho.flatten()).detach().cpu().numpy()
        std_2 = self.seq[2].rho_to_std(self.seq[2].weight_rho.flatten()).detach().cpu().numpy()
        self.plot_hist(mu_1, 'Layer 1: mean', "mu_1")
        self.plot_hist(mu_2, 'Layer 2: mean', "mu_2")
        self.plot_hist(std_1, 'Layer 1: std', "std_1")
        self.plot_hist(std_2, 'Layer 2: std', "std_2")

    def configure_optimizers(self):
        optimizer = torch.optim.Adagrad(self.parameters(), lr=self.train_params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, verbose=True)
        return [optimizer], [scheduler]

class Model_regime_1(BNN):
    def __init__(self, in_size, out_size, N, p, dist_params, train_params):
        super(Model_regime_1, self).__init__(in_size, out_size, N, p, dist_params, train_params, 1)

class Model_regime_2(BNN):
    def __init__(self, in_size, out_size, N, p, dist_params, train_params):
        dist_params['sigma_prior'] =  dist_params['sigma_prior'] * np.sqrt(N / (train_params['alpha'] * p))
        super(Model_regime_2, self).__init__(in_size, out_size, N, p, dist_params, train_params, 2)

class Model_regime_3(BNN):
    def __init__(self, in_size, out_size, N, p, dist_params, train_params):
        super(Model_regime_3, self).__init__(in_size, out_size, N, p, dist_params, train_params, 3)


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


class CNN(pl.LightningModule):
    def __init__(self, in_size, out_size, N, criterion, lr):
        super(CNN, self).__init__()
        self.save_hyperparameters()
    
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(6272, 128),
            nn.ReLU(),
            nn.Linear(128, 10))

        self.accuracy = torchmetrics.Accuracy()
        self.criterion = criterion
        self.lr = lr
        self.N = N

    def forward(self, x):
        pass

    def step(self, batch, batch_idx):
        x, y = batch
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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer




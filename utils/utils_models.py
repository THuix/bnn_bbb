import torchmetrics
from torch import nn
import numpy as np

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
    def __init__(self, dist_params, train_params, model_params, regime):
        super(BNN, self).__init__()
        if train_params['save_acc']:
            self.accuracy = torchmetrics.Accuracy()
            self.ECE = torchmetrics.CalibrationError(n_bins=15, norm='l1')

    def get_temperature(self, regime):
        if regime == 1 or regime == 2:
            return 1
        elif regime == 3:
            return self.train_params['alpha'] * self.train_params['p'] / self.model_params['w']
        else:
            raise ValueError('To implement')

    def check_params(self, params, true_params):
        for p in true_params:
            if not(p in params):
                raise ValueError(f"{p} is missing in {params}")
        return params

    def extract_flattened_weights(self):
        mu, std = [], []
        for module in self.seq:
            if hasattr(module, 'weight_mu'):
                mu.append(module.weight_mu.detach().cpu().flatten().tolist())
                std.append(module.rho_to_std(module.weight_rho).detach().cpu().flatten().tolist())
        return np.concatenate(mu), np.concatenate(std)

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
        pred = self.seq(x) / self.model_params['N_last_layer']
        nll = self.train_params['criterion'](pred, y) / self.T
        kl = self._get_kl() / self.train_params['nb_batches']
        obj_loss = nll + kl
        return obj_loss, nll, kl, pred

    def re_balance_loss(self, loss):
        if self.regime == 1 or self.regime == 3: 
            return self.train_params['alpha'] * loss * self.train_params['nb_batches'] / self.model_params['w']
        elif self.regime == 2:
            return loss * self.train_params['nb_batches'] / self.train_params['p']
        else:
            raise ValueError('To implement')

    def step(self, batch, batch_idx):
        x, y = batch
        if self.do_flatten:
            x = x.reshape(x.size()[0], -1)
        obj_loss = torch.zeros(1, requires_grad=True).type_as(x)
        nll = torch.zeros(1).type_as(x)
        kl = torch.zeros(1).type_as(x)
        pred = torch.zeros((x.size()[0], self.model_params['out_size'])).type_as(x)

        for idx in range(self.train_params['nb_samples']):
            o, n, k, p = self._step_1_sample(x, y)
            obj_loss = obj_loss + o / self.train_params['nb_samples']
            nll = nll + n / self.train_params['nb_samples']
            kl = kl + k / self.train_params['nb_samples']
            pred = pred + p / self.train_params['nb_samples']
        
        obj_loss = self.re_balance_loss(obj_loss)
        nll = self.re_balance_loss(nll)
        kl = self.re_balance_loss(kl)

        
        
        logs = {
            "obj": obj_loss,
            "kl": kl,
            "nll": nll,
            "ratio_nll_kl": nll / kl}

        if self.train_params['save_acc']:
            self.accuracy.update(pred, y)
            self.ECE.update(pred, y)
            logs['acc'] = self.accuracy.compute()
            logs['ece'] = self.ECE.compute()

        
        return obj_loss, logs   

    def training_epoch_end(self, output):
        mu, std = self.extract_flattened_weights()
        self.log_dict({
            'max_mu': np.max(mu),
            'min_mu': np.min(mu),
            'max_std': np.max(std),
            'min_std': np.min(std),
            'mean_mu': np.mean(mu),
            'mean_std': np.mean(std),
            'median_mu': np.median(mu),
            'median_std': np.median(std)
        }, sync_dist=True)
    
    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, sync_dist=True)
        self.accuracy.reset()
        self.ECE.reset()
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        self.accuracy.reset()
        self.ECE.reset()
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
        self.accuracy.reset()
        self.ECE.reset()
        return loss

    def test_epoch_end(self, output):
        self.mu, self.std = self.extract_flattened_weights()
        self.ratio = self.mu / self.std
        self.samples = np.random.normal(self.mu, self.std)
        self.plot_hist(self.mu, 'Mean', "mu")
        self.plot_hist(self.std, 'Std', "std")
        self.plot_hist(self.ratio, 'ratio mu / std', 'ratio_weights')
        self.plot_hist(self.samples, 'weights', 'weights')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.train_params['lr'])
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.train_params['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20 , gamma=0.1, verbose=True)
        return [optimizer], [scheduler]






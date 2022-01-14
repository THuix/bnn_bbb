import torch
from torch import nn
import torch.nn.functional as F
from distributions import norm_log_prob
from security import check
import numpy as np

class Linear_bnn(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 init_rho_post,
                 init_mu_post,
                 sigma_prior,
                 mu_prior,
                 N,
                 p,
                 alpha,
                 init_type='fixed',
                 regime=0,
                 bias = False):
        """
        [args]:
            - in_size(integer): input size of the layer
            - out_size(integer): output size of the layer
            - init_rho_post(float): initialization rho for the variational posterior
            - init_mu_post(float): initialization mu for the variational posterior
            - sigma_prior(float): standard deviation of the prior
            - mu_prior(float): mean of the prior
            - N(integer): number of neurons
            - p(integer): number of data points
            - alpha(float): 
            - init_type([fixed, normal]): strategy for the weights initialization
            - regime([1, 2, 3]): regime used
            - bias(Boolean): add bias to the layer        
        """
        super(Linear_bnn, self).__init__()
        self.N = N
        self.p = p
        self.alpha = alpha
        self.regime = regime
        self.bias = bias
        self.mu_prior = mu_prior
        self.sigma_prior = sigma_prior

        self.weight_mu = self.init_parameter(init_mu_post, init_type, (out_size, in_size))
        self.weight_rho = self.init_parameter(init_rho_post, init_type, (out_size, in_size))

        if bias:
            self.bias_mu = self.init_parameter(init_mu_post, init_type, (out_size,))
            self.bias_rho = self.init_parameter(init_mu_post, init_type, (out_size,))

    def init_parameter(self, init_value, init_type, size):
        """
        [args]
            - init_value(float): initialization value
            - init_type([fixed, normal]): initialization strategy
            - size(tupe): tuple representing the size of the layer weights

        [objective]
        This method initializes the weights with a given strategy
        """
        if init_type == 'fixed':
            return nn.Parameter(torch.ones(size) * init_value)
        elif init_type == 'normal':
            return nn.Parameter(torch.normal(init_value, 0.1, size=size))
        else:
            raise ValueError('To implement')

    def rho_to_std(self, rho):
        """
        [args]
            - rho(float): rho matrix

        [objective]
        This method converts the rho matrix to the standard deviation matrix (sigma) using the
        reparametrization function 
        """
        if self.regime == 1 or self.regime == 3:
            value = torch.where(rho < 50, torch.log1p(torch.exp(rho)), rho)

        elif self.regime == 2:
            value = torch.log(torch.exp(self.sigma_prior + rho) + 1)
            # f_inf = f_inf = torch.sqrt(self.sigma_prior**2 * (torch.log((self.sigma_prior**2) / torch.log1p(torch.exp(rho))**2) + (torch.log1p(torch.exp(rho))**2) / (self.sigma_prior**2) - 1) / np.sqrt(self.alpha * self.p))
            # kernel = 1 / (1 + self.N * (rho - np.log(np.exp(self.sigma_prior))**2))
            # value = self.sigma_prior * np.sqrt(self.N / (self.p * self.alpha)) * (1 - kernel) + f_inf
        else:
            raise ValueError('To implement')
        check(value, items=(rho, self.N, self.p))
        return value

    def sample(self, mu, rho):
        """
        [args]
            - mu(float): mean of the distribution
            - rho(float): rho of the distribution

        [objective]
        This method returns a sample from the gaussian distribution(mu, rho)
        """
        eps = torch.normal(0., 1., size=mu.size()).type_as(mu)
        value = mu + self.rho_to_std(rho) * eps
        check(value)
        return value

    def forward(self, x):
        """
        [args]

        [objective]
        """
        w = self.sample(self.weight_mu, self.weight_rho)
        log_var_post = norm_log_prob(w, self.weight_mu, self.rho_to_std(self.weight_rho))
        log_prior = norm_log_prob(w, self.mu_prior, self.sigma_prior)
        self.appro_kl = log_var_post - log_prior
        check(self.appro_kl)

        if self.bias:
            b = self.sample(self.bias_mu, self.bias_rho)
            log_var_post = norm_log_prob(b, self.bias_mu, self.rho_to_std(self.bias_rho))
            check(log_var_post)
            log_prior = norm_log_prob(b, self.mu_prior, self.sigma_prior)
            check(log_prior)
            self.appro_kl += log_var_post - log_prior
            check(self.appro_kl)
            out =  F.linear(x, w, b)
        else:
            out = F.linear(x, w, None)
        check(out)
        return out

from utils_run_exp import main, load_data, get_model, get_trainer
import argparse
import numpy as np
from torch import nn
import wandb
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import pytorch_lightning as pl
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--model_name')
parser.add_argument('--range_N', default=[], nargs='+')
parser.add_argument('--range_alpha', default=[], nargs='+')
parser.add_argument('--nb_epochs', type=int)  
parser.add_argument('--project_name')
parser.add_argument('--p_scales_with_N', type=bool)
parser.add_argument('--lr', type=float)
parser.add_argument('vgg_type', default=None)

num_works = 0
batch_size = 128

def init_model_with_sgd(model, nn_model):
    nn_modules = list(nn_model.seq)
    for idx, module in enumerate(model.seq):
        if hasattr(module, 'weight_mu'):
            model.weight_mu = nn_modules[idx].weight.data
    return model

def main_for_vgg(project_name, model_name, dataset_name, num_works, batch_size, dist_params, train_params, model_params, nn_model):
    trainset, testset = load_data(batch_size, dataset_name, num_works, train_params, model_params)
    model = get_model(model_name, dist_params, train_params, model_params)
    model = init_model_with_sgd(model, nn_model)
    exp_name = 'init_sgd'
    wandb_logger = WandbLogger(name=exp_name,project=project_name)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = get_trainer(train_params['nb_epochs'], wandb_logger, lr_monitor, exp_name)
    trainer.fit(model, trainset, testset)
    result = trainer.test(model, testset)
    wandb.finish()
    return model


def train_nn(args):

    dist_params = {}

    train_params = {'lr': 0.01,
                    'nb_epochs': args.nb_epochs,
                    'criterion': nn.CrossEntropyLoss(reduction='mean'),
                    'dataset': args.dataset,
                    'model': 'VGG_classic',
                    'alpha': None}

    train_params['limit_p'] = None

    model_params = {'padding' : 1,
                    'dilation': 1,
                    'stride': 1,
                    'kernel_size': 3}

    model_params['VGG_type'] = int(args.vgg_type)

    return main(args.project_name,
                train_params['model'],
                args.dataset,
                num_works,
                batch_size,
                dist_params,
                train_params,
                model_params)

if __name__ == '__main__':
    wandb.finish()
    args = parser.parse_args()

    model_nn = train_nn(args)
    

    for alpha in args.range_alpha:

            dist_params = {'init_mu_post': 0.,
                           'init_rho_post': np.log(np.exp(0.1)-1),
                           'sigma_prior': 0.1,
                           'mu_prior': 0.}

            if alpha != 'None':
                alpha = float(alpha)

            train_params = {'lr': args.lr,
                             'nb_epochs': args.nb_epochs,
                            'nb_samples': 1,
                            'criterion': nn.CrossEntropyLoss(reduction='sum'),
                            'alpha': alpha,
                            'dataset': args.dataset,
                            'model': args.model_name}

            train_params['limit_p'] = None

            model_params = {'padding' : 1,
                            'dilation': 1,
                            'stride': 1,
                            'kernel_size': 3}

            model_params['VGG_type'] = int(args.vgg_type)

            main_for_vgg(args.project_name,
                 args.model_name,
                 args.dataset,
                 num_works,
                 batch_size,
                 dist_params,
                 train_params,
                 model_params,
                 model_nn)
 
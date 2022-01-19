from utils_run_exp import main
import argparse
import numpy as np
from torch import nn
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--range_alpha', default=[], nargs='+')
parser.add_argument('--nb_epochs', type=int)  
parser.add_argument('--project_name')

num_works = 8
batch_size = 128

def launch_train(args, alpha, model_name):
    train_params = {'lr': 1e-2,
                        'nb_epochs': args.nb_epochs,
                        'nb_samples': 3,
                        'criterion': nn.CrossEntropyLoss(reduction='sum'),
                        'alpha': alpha}

    model_params = {}

    dataset_name = args.dataset

    main(args.project_name,
        model_name,
        args.dataset,
        num_works,
        batch_size,
        dist_params,
        train_params,
        model_params)


if __name__ == '__main__':
    wandb.finish()
    args = parser.parse_args()
    dist_params = {'init_mu_post': 0.,
                    'init_rho_post': np.log(np.exp(0.1)-1),
                    'sigma_prior': 0.1,
                    'mu_prior': 0.}

    launch_train(args, None, 'Resnet_regime_1')

    for alpha in args.range_alpha:
        alpha = float(alpha)
        launch_train(args, alpha, 'Resnet_regime_3')

        
 
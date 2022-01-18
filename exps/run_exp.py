from utils_run_exp import main
import argparse
import numpy as np
from torch import nn
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--model_name')
parser.add_argument('--range_N', default=[], nargs='+')
parser.add_argument('--range_alpha', default=[], nargs='+')
parser.add_argument('--nb_epochs', type=int)  

num_works = 8
batch_size = 128

if __name__ == '__main__':
    wandb.finish()
    args = parser.parse_args()
    dist_params = {'init_mu_post': 0.,
                    'init_rho_post': np.log(np.exp(0.1)-1),
                    'sigma_prior': 0.1,
                    'mu_prior': 0.}

    for N_last_layer in args.range_N:
        for alpha in args.range_alpha:

            if alpha != 'None':
                alpha = float(alpha)

            train_params = {'lr': 1e-2,
                             'nb_epochs': args.nb_epochs,
                            'nb_samples': 3,
                            'criterion': nn.MSELoss(reduction='sum') if args.dataset == 'BOSTON' else nn.CrossEntropyLoss(reduction='sum'),
                            'alpha': alpha}

            model_params = {'padding' : 0,
                            'dilation': 1,
                            'stride': 1,
                            'kernel_size': 3,
                            'N_last_layer': int(N_last_layer)}

            nb_epochs = args.nb_epochs
            model_name = args.model_name
            dataset_name = args.dataset
            project_name = f'{model_name}_nb_epochs_{nb_epochs}_{dataset_name}'

            main(project_name,
                 args.model_name,
                 args.dataset,
                 num_works,
                 batch_size,
                 dist_params,
                 train_params,
                 model_params)
 
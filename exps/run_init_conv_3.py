
from run_exp_conv import main_conv
import argparse
import torch
import numpy as np

lr = 1e-2
nb_samples = 10
init_rho_post = np.log(np.exp(1.)-1)
regime = 3

criterion = torch.nn.CrossEntropyLoss(reduction='sum')
nb_epochs = 0
parser = argparse.ArgumentParser()
parser.add_argument('dataset')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataset == '0':
        print('CIFAR10 DATASET')
        dataset_name = "CIFAR10"
    elif args.dataset == "1":
        print('MNIST dataset')
        dataset_name = 'MNIST'
    else:
        print('BOSTON')
        dataset_name = 'BOSTON'

    range_param = range(10, 100, 10)
    alpha = 1 / 60000

    project_name = f'bnn_bbb_regime_{regime}_init_{dataset_name}_conv'
    for hidden_channels in range_param:
        main_conv(hidden_channels, lr, nb_samples, alpha, regime, project_name, dataset_name, criterion, nb_epochs, init_rho_post)
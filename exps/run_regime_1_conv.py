from run_exp_conv import main_conv
import argparse
import torch
import numpy as np

lr = 1e-2
nb_samples = 3
alpha = None
init_rho_post = np.log(np.exp(1.)-1)
regime = 1
nb_epochs = 10
criterion = torch.nn.CrossEntropyLoss(reduction='sum')

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == '0':
        print('CIFAR10 DATASET')
        dataset_name = "CIFAR10"
    else:
        print('MNIST dataset')
        dataset_name = 'MNIST'

    range_N = [100]
    project_name = f'conv_bnn_bbb_regime_1_{dataset_name}'
    for N in range_N:
        main_conv(N, lr, nb_samples, alpha, 1, project_name, dataset_name, criterion, nb_epochs, init_rho_post)
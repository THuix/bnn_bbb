
from run_exp import main
import argparse
import torch


lr = 1e-2
nb_samples = 3
regime = 3
criterion = torch.nn.MSELoss(reduction='sum')
nb_epochs = 0
limit_train_batches = 1.

parser = argparse.ArgumentParser()
parser.add_argument('dataset')

if __name__ == '__main__':
    args = parser.parse_args()

    dataset_name = 'BOSTON'

    # if args.dataset == '0':
    #     print('CIFAR10 DATASET')
    #     dataset_name = "CIFAR10"
    # else:
    #     print('MNIST dataset')
    #     dataset_name = 'MNIST'

    range_N = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    alpha = 1/60000

    project_name = f'new_bnn_bbb_regime_{regime}_init_{dataset_name}'
    for N in range_N:
        main(N, lr, nb_samples, alpha, regime, project_name, dataset_name, criterion, nb_epochs, limit_train_batches)
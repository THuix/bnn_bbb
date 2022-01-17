
from run_exp import main
import argparse
import torch


lr = 1e-2
nb_samples = 100
regime = 2
#criterion = torch.nn.MSELoss(reduction='sum')
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

    range_N = range(100, 1000, 100)
    alpha = 10000 / 60000

    project_name = f'new_bnn_bbb_regime_{regime}_init_{dataset_name}'
    for N in range_N:
        main(N, lr, nb_samples, alpha, regime, project_name, dataset_name, criterion, nb_epochs)
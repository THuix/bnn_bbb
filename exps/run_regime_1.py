from run_exp import main
import argparse
import torch

lr = 1e-1
nb_samples = 3
alpha = None
regime = 1
nb_epochs = 100
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
limit_train_batches = 2

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

    range_N = [1000]
    project_name = f'new_bnn_bbb_regime_1_{dataset_name}'
    for N in range_N:
        main(N, lr, nb_samples, alpha, 1, project_name, dataset_name, criterion, nb_epochs, limit_train_batches)
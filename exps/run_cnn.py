from run_exp import main
import argparse
import torch

lr = 1e-1
regime = "cnn"
p = 60000
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
nb_epochs = 100

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

    range_N = [128]
    project_name = f'bnn_bbb_regime_cnn_{dataset_name}'
    for N in range_N:
        main(N, lr, None, None, 'cnn', project_name, dataset_name, criterion, nb_epochs, p)
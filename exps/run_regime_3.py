
from run_exp import main
import argparse

lr = 1e-1
nb_samples = 3
regime = 3
N = 500

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

    range_alpha = [1/6, 1/60, 1/600, 1/6000, 1/60000, 1/600000, 1/6000000, 1/60000000]
    project_name = f'bnn_bbb_regime_1_{args.dataset}'
    for alpha in range_alpha:
        main(N, lr, nb_samples, alpha, regime, project_name, dataset_name)
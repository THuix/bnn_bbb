from run_exp import main
import argparse

lr = 1e-1
nb_samples = 3
alpha = None
regime = 1

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

    range_N = [100] #, 20, 30, 40, 50, 60, 70, 80, 90]
    project_name = f'bnn_bbb_regime_1_{dataset_name}'
    for N in range_N:
        main(N, lr, nb_samples, alpha, regime, project_name, dataset_name)
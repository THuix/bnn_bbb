from run_exp import main
import argparse

lr = 1e-1
regime = "nn"
p = 60000
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

    range_N = [50000]
    project_name = f'bnn_bbb_regime_nn_{dataset_name}'
    for N in range_N:
        main(N, lr, None, None, 'nn', project_name, dataset_name, p)
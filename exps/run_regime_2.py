from run_exp import main
import argparse

lr = 1e-1
regime = 2
N = 1000
sigma_prior = 1.

parser = argparse.ArgumentParser()
parser.add_argument('dataset')
if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 0:
        dataset_name = "CIFAR10"
    else:
        dataset_name = 'MNIST'

    range_alpha = [0.013, 0.008, 0.01, 0.005]
    project_name = f'bnn_bbb_regime_1_{args.dataset}'
    for alpha in range_alpha:
        nb_samples = int(sigma_prior**2 * N / (alpha * 60000))
        main(N, lr, nb_samples, alpha, regime, project_name)
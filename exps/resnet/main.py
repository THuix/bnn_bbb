import torch
import os
import sys
import numpy as np
from torch import nn
import torchmetrics
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import Compose 
from scipy.special import softmax
from tqdm import tqdm
sys.path.insert(0,'../../utils')
from models import Resnet_regime_3
from model_resnet import Resnet20_classic
import pickle as pkl


def load_models():
    device = torch.device('cuda')
    model_1 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_1.ckpt").to(device)
    model_01 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_01.ckpt").to(device)
    model_001 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_001.ckpt").to(device)
    model_10 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_10.ckpt").to(device)
    model_100 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_100.ckpt").to(device)
    #model_500 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_500.ckpt").to(device)
    model_1000 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_1000.ckpt").to(device)
    model_0001 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_0001.ckpt").to(device)

    nn_1 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_1.ckpt").to(device)
    nn_01 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_01.ckpt").to(device)
    nn_001 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_001.ckpt").to(device)
    nn_10 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_10.ckpt").to(device)
    nn_100 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_100.ckpt").to(device)
    #nn_500 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/nn_500.ckpt").to(device)
    nn_1000 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_1000.ckpt").to(device)
    nn_0001 = Resnet20_classic.load_from_checkpoint("../../exps/resnet/nn_0001.ckpt").to(device)


    models = [(0.001, model_0001, nn_0001),
        (0.01, model_001, nn_001),
         (0.1, model_01, nn_01),
         (1., model_1, nn_1),
         (10., model_10, nn_10),
         (100, model_100, nn_100),
         (1000, model_1000, nn_1000)]
        
    return models, device

def load_dataset():
    batch_size = 200
    num_works= 0
    test_transform = Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    val_dataset = CIFAR10('../', download=False, transform=test_transform, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    return val_loader

ECE = torchmetrics.CalibrationError(n_bins=15, norm='l1').to('cuda')
accuracy = torchmetrics.Accuracy().to('cuda')
criterion = nn.CrossEntropyLoss(reduction='mean')

def compute(model, dataset, device, nb_samples):
    acc, ece, nll, conf = 0, 0, 0, 0
    batch_size = 200
    results = torch.empty(len(dataset), batch_size, nb_samples, 50)
    labels = torch.empty(len(dataset), batch_size)
    for batch_idx, (x, y) in enumerate(dataset):
        x = x.to(device)
        y = y.to(device)
        labels[batch_idx, :] = y.detach().cpu()
        for idx in range(nb_samples):
            results[batch_idx, :, idx, :] = model(x).detach().cpu()
    return results, labels
    
if __name__ == '__main__':
    val_loader = load_dataset()
    models, device = load_models()
    results_list, results_nn_list, eta_list = [], [], []

    for eta, model, model_nn in tqdm(models):
        print('[SYSTEM]', eta)
        results, labels = compute(model, val_loader, device, 15)
        del model
        results_nn, labels_nn = compute(model_nn, val_loader, device, 1)
        del model_nn
        eta_list.append(eta)
        results_nn_list.append((results_nn, labels_nn))
        results_list.append((results, labels))
    results = {'eta_list': eta_list,
                'results_list': results_list,
                'results_nn_list': results_nn_list}
    pkl.dump(results, open('results.pkl', 'wb'))


    


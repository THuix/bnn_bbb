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

batch_size = 200

def load_models():
    device = torch.device('cuda')
    model_1 = Resnet_regime_3.load_from_checkpoint("resnet_1.ckpt").to(device)
    model_01 = Resnet_regime_3.load_from_checkpoint("resnet_01.ckpt").to(device)
    model_001 = Resnet_regime_3.load_from_checkpoint("resnet_001.ckpt").to(device)
    model_10 = Resnet_regime_3.load_from_checkpoint("resnet_10.ckpt").to(device)
    model_100 = Resnet_regime_3.load_from_checkpoint("resnet_100.ckpt").to(device)
    #model_500 = Resnet_regime_3.load_from_checkpoint("../../exps/resnet/bnn_500.ckpt").to(device)
    model_1000 = Resnet_regime_3.load_from_checkpoint("resnet_1000.ckpt").to(device)
    model_0001 = Resnet_regime_3.load_from_checkpoint("resnet_0001.ckpt").to(device)

    sgd = Resnet20_classic.load_from_checkpoint("resnet_nn.ckpt").to(device)

    models = [(0.001, model_0001),
        (0.01, model_001),
         (0.1, model_01),
         (1., model_1),
         (10., model_10, nn_10),
         (100, model_100, nn_100),
         (1000, model_1000)]
        
    return models, device, sgd

def load_dataset():
    num_works= 0
    test_transform = Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
    val_dataset = CIFAR10('../', download=True, transform=test_transform, train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
    return val_loader

ECE = torchmetrics.CalibrationError(n_bins=15, norm='l1').to('cuda')
accuracy = torchmetrics.Accuracy().to('cuda')
criterion = nn.CrossEntropyLoss(reduction='mean')

def compute(model, dataset, device, nb_samples):
    acc, ece, nll, conf = 0, 0, 0, 0
    results = torch.empty(len(dataset), batch_size, nb_samples, 10)
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
    models, device, sgd = load_models()
    results_list, eta_list = [], []

    for idx in range(5):
        print(idx)
        for eta, model in tqdm(models):
            print('[SYSTEM]', eta)
            results, labels = compute(model, val_loader, device, 20)
            del model
            eta_list.append(eta)
            results_list.append((results, labels))

        result_nn = compute(sgd, val_loader, device, 1)
        results = {'eta_list': eta_list,
                    'results_list': results_list,
                    'nn': result_nn}
        pkl.dump(results, open(f'results_{idx}.pkl', 'wb'))


    


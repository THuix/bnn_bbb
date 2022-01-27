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
    batch_size = 500
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
    for x, y in dataset:
        accuracy.reset()
        ECE.reset()
        x = x.to(device)
        y = y.to(device)
        pred = torch.empty(nb_samples, x.size()[0], 10).to(device)
        for idx in range(nb_samples):
            pred[idx, :] = model(x).softmax(dim=1)
            
        avg_pred = pred.mean(dim=0)
        pred_prob = avg_pred[range(avg_pred.size()[0]), y]
        nll += - torch.log(pred_prob).mean().item()
        
        print(avg_pred.device, y.device)
        accuracy.update(avg_pred, y)
        acc += accuracy.compute().item()
        
        ECE.update(avg_pred, y)
        ece += ECE.compute().item()
        
        conf += pred_prob.mean().item()
        
        ECE.reset()
        accuracy.reset()
    return acc/len(dataset), ece / len(dataset), nll / len(dataset), conf / len(dataset)
    

def plot(x, y, y_nn, title, savefile):
    plt.plot(x, y, '-o', label='BNN')
    plt.plot(x, y, '-o', label='Baseline SGD')
    plt.title(title)
    plt.xscale('log')
    plt.legend()
    plt.savefig(savefile)
    plt.close()

def plot_curves(eta_list, ece_list, ece_list_nn, acc_list, acc_list_nn, nll_list, nll_list_nn, p_list, p_list_nn):
    plot(eta_list, ece_list, ece_list_nn, 'ECE', 'ece.pdf')
    plot(eta_list, acc_list, acc_list_nn, 'Accuracy', 'acc.pdf')
    plot(eta_list, nll_list, nll_list_nn, 'Negative Log likelihood', 'nll.pdf')
    plot(eta_list, p_list, p_list_nn, 'True probability', 'p.pdf')

if __name__ == '__main__':
    val_loader = load_dataset()
    models, device = load_models()
    eta_list, acc_list, ece_list, nll_list, p_list = [], [], [], [], []
    eta_list_nn, acc_list_nn, ece_list_nn, nll_list_nn, p_list_nn = [], [], [], [], []
    for eta, model, model_nn in tqdm(models):
        acc, ece, nll, p = compute(model, val_loader, device, 100)
        acc_nn, ece_nn, nll_nn, p_nn = compute(model_nn, val_loader, device, 1)
        eta_list.append(eta)
        ece_list.append(ece); ece_list_nn.append(ece_nn)
        acc_list.append(acc); acc_list_nn.append(acc_nn)
        nll_list.append(nll); nll_list_nn.append(nll_nn)
        p_list.append(p); p_list_nn.append(p_nn)
    plot_curves(eta_list,
                ece_list,
                ece_list_nn,
                acc_list, 
                acc_list_nn,
                nll_list,
                nll_list_nn,
                p_list,
                p_list_nn)


    


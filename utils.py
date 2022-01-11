from security import check, check_parameters, check_param_grad
from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy

def full_training(nb_epochs, dataset, optim, model, criterion, nb_samples, device, print_step, scheduler=None, save_weights=False):
    results = torch.empty(nb_epochs, 3)
    weights = {}
    for epoch in range(nb_epochs):
        train_1_epoch(dataset, optim, model, criterion, nb_samples, device, scheduler)
        loss, nll, kl = evaluate(dataset, model, criterion, nb_samples, device)
        results[epoch, :] = torch.tensor([loss, nll, kl])
        if print_step != None and epoch % print_step == 0:
           print(f"Epoch = {epoch} | Obj Loss = {loss} | KL = {kl} | NLL = {nll}")
        if save_weights:
            weights[epoch] = [deepcopy(param.data.detach().numpy().reshape(-1)) for param in model.parameters()]
    return model, results, weights

def train_1_epoch(dataset, optim, model, criterion, nb_samples, device, scheduler=None):
    model.train()
    for X, y in dataset:
        optim.zero_grad()
        check_parameters(model)
        obj_loss, _, _ = sample_from_model(X, y, nb_samples, device, model, criterion)
        obj_loss.backward()
        check_param_grad(model)
        optim.step() 
    if scheduler != None: 
        try:
            scheduler.step() 
        except:
            scheduler.step(obj_loss)

def sample_from_model(X, y, nb_samples, device, model, criterion):
    nll, kl = 0, 0
    for _ in range(nb_samples):
        pred = model(X)
        check(pred)
        nll += criterion(pred, y)
        check(nll, items=(pred, y))
        kl += model.get_kl()
        check(kl) 
    nll = nll / nb_samples
    kl = kl / nb_samples
    obj_loss = nll + kl
    check(obj_loss)
    return obj_loss, nll, kl

# def sample_from_model(X, y, nb_samples, device, model, criterion):
#     for idx in range(nb_samples):
#         if idx == 0:
#             prediction = model(X) / nb_samples
#             kl = model.get_kl() / nb_samples
#         else:
#             prediction += model(X) / nb_samples
#             kl += model.get_kl() / nb_samples
#         check(prediction)
#         check(kl)         
#     nll = criterion(prediction, y) 
#     check(nll, items=(prediction, y))
#     obj_loss = nll + kl
#     check(obj_loss)
#     return obj_loss, nll, kl
    
def sample_prediction(X, y, nb_samples, model, pred_size):
    pred = torch.zeros(pred_size)
    for _ in range(nb_samples):
        pred += model(X)
    return pred / nb_samples

def evaluate(dataset, model, criterion, nb_samples, device):
    model.eval()
    with torch.no_grad():
        mean_loss, mean_nll, mean_kl = 0, 0, 0
        for X, y in dataset:
            obj_loss, nll, kl = sample_from_model(X, y, nb_samples, device, model, criterion)
            mean_loss += obj_loss.item() / len(dataset)
            mean_nll += nll.item() / len(dataset)
            mean_kl += kl.item() / len(dataset)
    return mean_loss, mean_nll, mean_kl

def compute_accuracy(X, y, model, nb_samples, pred_size):
    pred = sample_prediction(X, y, nb_samples, model, pred_size)
    check(pred)
    acc = np.mean(np.where((torch.argmax(pred, dim=1) == y).cpu().numpy(), 1, 0))
    return acc
    
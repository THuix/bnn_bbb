import torch

def check(tensor, msg="", items=None):
    error = False
    if torch.any(torch.isnan(tensor)):
        msg += "Nan Value\n"
        error = True
    if torch.any(torch.isinf(tensor)):
        msg += "Inf Value\n"
        error = True
    if error:
        raise ValueError(msg, items)

def check_parameters(model):
    for idx, param in enumerate(model.parameters()):
        check(param, f"Parameter {idx}\n")

def check_param_grad(model):
    for idx, param in enumerate(model.parameters()):
        check(param.grad, f"Parameter {idx}\n")

def check_is_tensor(data, name):
    if not(isinstance(data, torch.Tensor)):
        raise ValueError(f'{name} is not a torch tensor')
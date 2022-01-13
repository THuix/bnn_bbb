import torch
import numpy as np
from security import check

eps = 1e-6

def norm_log_prob(w, m, s):
        if type(s) == torch.Tensor:
            s[s==0.] = eps
        else:
            if s == 0.:
                s = eps
        log_prob = torch.distributions.normal.Normal(m, s).log_prob(w)
        check(log_prob, items=(w, m, s))
        return log_prob.sum()

def exact_kl(m1, s1, m2, s2):
    return torch.log(s2/s1) + (s1**2 + (m1-m2)**2) / (2*s2**2) -0.5
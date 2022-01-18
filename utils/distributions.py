import torch
import numpy as np
from security import check

eps = 1e-6

def norm_log_prob(w, m, s):
        # if type(s) == torch.Tensor:
        #     s[s==0.] = eps
        # else:
        #     if s == 0.:
        #         s = eps
        # log_prob = torch.distributions.normal.Normal(m, s).log_prob(w)

        print(w.size(), type(m))
        p = torch.exp(- (w - m)**2 / (2 * s**2)) / (s * np.sqrt(2 * np.pi))
        log_prob = torch.log(p)
        check(log_prob, items=(w, m, s))
        return log_prob.sum()

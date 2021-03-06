import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler

class BostonDataset(torch.utils.data.Dataset):
    '''
    Prepare the Boston dataset for regression
    '''

    def __init__(self):
        X, y = load_boston(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        self.data = torch.from_numpy(X).float()
        self.targets = torch.from_numpy(y).float()
            
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]
    
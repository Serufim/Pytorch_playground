import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X_train = X
        self.y_train = y
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_train, self.y_train = self.X_train.astype(np.float32), self.y_train.astype(np.long)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X_train[idx]), torch.tensor(self.y_train[idx], dtype=torch.long)

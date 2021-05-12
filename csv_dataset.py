import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CSVDataset(Dataset):
    def __init__(self, file_name):
        dataframe = pd.read_csv(file_name)
        X_train = dataframe.iloc[:, :-1]
        y_train = dataframe.iloc[:, -1]
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)

        self.X_train = torch.Tensor(X_train, dtype=torch.float32)
        self.y_train = torch.Tensor(y_train)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

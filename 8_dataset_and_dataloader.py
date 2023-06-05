from typing import Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class WineDs(Dataset):
    def __init__(self, dataset_path) -> None:
        super().__init__()

        self.ds_raw = pd.read_csv(dataset_path, names=np.arange(0, 13)).to_numpy()
        self.ds = torch.from_numpy(self.ds_raw)

        self.x = self.ds[:, :-1]
        self.y = self.ds[:, -1]

        self.length = self.ds.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]

    def __len__(self):
        return self.length


batch_size = 16
epochs = 100

ds_path = "/home/neotod/compsci/datasets/wine.csv"
ds = WineDs(ds_path)

data_loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=2)
data_iter = iter(data_loader)

for ep in range(epochs):
    for i, (x, y) in enumerate(data_loader):
        print(i)

    print("finished")


# dataset transform
class ToTensor:
    def __call__(self, sample):
        x, y = sample

        return torch.from_numpy(x), torch.from_numpy(y)


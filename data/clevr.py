import os
import torch
import h5py as h5
from torch.utils.data import Dataset


class CLEVR(Dataset):
    def __init__(self, dataset_path, mode="train"):
        self.dataset_path = dataset_path
        self.mode = mode

    def path(self):
        return os.path.join(self.base_path, "images", self.mode)

    def __len__(self):
        with open(self.dataset_path, "rb") as f:
            data = h5.File(f)["images"][self.mode]
            return data.shape[0]

    def __getitem__(self, idx):
        with open(self.dataset_path, "rb") as f:
            data = h5.File(f)["images"][self.mode]

            return torch.tensor(data[idx])

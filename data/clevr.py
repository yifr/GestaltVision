import os
import torch
from PIL import Image
from torch.utils.data import Dataset


class CLEVR(Dataset):
    def __init__(self, base_path, train_test_split=0.8, mode="train"):
        self.base_path = base_path
        self.train_test_split = train_test_split
        self.mode = mode

    def path(self):
        return os.path.join(self.base_path, "images", self.mode)

    def __len__(self):
        return len(os.listdir(self.path()))

    def __getitem__(self, idx):
        img_name = f"CLEVR_{self.mode}_{idx:06d}.png"
        img_path = os.path.join(self.path(), img_name)
        image = Image.open(img_path).to("RGB")


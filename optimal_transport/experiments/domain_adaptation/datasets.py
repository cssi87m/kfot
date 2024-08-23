import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple


class OfficeDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        metadata_file: str,
        transform=None,
        **kwargs,
    ):
        self.data, self.target = [], []
        with open(metadata_file, "r") as f:
            for line in f:
                path, label = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                self.data.append(path)
                self.target.append(int(label))
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.data[index], self.target[index]
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
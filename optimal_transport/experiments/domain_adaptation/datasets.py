import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from typing import Tuple
import torchvision.transforms as T
from typing import Optional

from .utils import load_features


class OfficeDataset(Dataset):
    def __init__(
        self, 
        root_dir: str,
        annotation_file: str,
        transforms: Optional[T.Compose] = None,
        **kwargs,
    ):
        self.data, self.target = [], []
        with open(annotation_file, "r") as f:
            for line in f:
                path, label = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                self.data.append(path)
                self.target.append(int(label))
        self.transforms = transforms

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, target = self.data[index], self.target[index]
        img = Image.open(path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.data)
    
   
class FeatureDataset(Dataset):
    def __init__(
        self, 
        feature_file: str,
        annotation_file: str,
        **kwargs
    ):
        super().__init__()
        self.target = []
        with open(annotation_file, "r") as f:
            for line in f:
                _, label = line.strip().split(' ')
                self.target.append(int(label))
        self.feature = load_features(feature_file)
    
    def __len__(self) -> int:
        return len(self.feature)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.feature[index], self.target[index]
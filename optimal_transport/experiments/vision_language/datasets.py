import numpy as np 
import torch 
from torchvision.datasets import CIFAR10
import torchvision.transforms as TF
from torch.utils.data import Subset, DataLoader


class cifar10(CIFAR10): 
    def __init__(self, train = False, transform:TF = None, target_transform:TF = None):
        super().__init__('data/cifar10', train=train, transform=transform, target_transform=target_transform, download=True) 
        self.labels = [f"A photo of {self.classes[i]}" for i in self.targets]

        self.data_label = [(self.data[i], self.labels[i]) for i in range(len(self))]
    
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        return self.data[idx], self.labels[idx], self.targets[idx]
    
    def get_batch_loader(self, class_name, batch_size=32):
        indices = [i for i, target in enumerate(self.targets) if self.classes[target] == class_name]
        
        subset_data = Subset(self, indices)
        loader = DataLoader(subset_data, batch_size=batch_size, shuffle=False)
        
        return loader

    
# test = cifar10() 
# print(test[0])
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from collections import OrderedDict
import torch


class ChenXiMLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(ChenXiMLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i+1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i+1)] = nn.ReLU()
            layers['drop{}'.format(i+1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model= nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)
    

MODEL_URLS = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

def mnist_chenxi_mlp(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None) -> nn.Module:
    model = ChenXiMLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(MODEL_URLS['mnist'], map_location=torch.device('cpu'))
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model
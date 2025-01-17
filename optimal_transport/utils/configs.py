import yaml
import json
import os
import sys
import copy
from torch import nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, Any, Union, Tuple, Callable

# for config abbreviation
import torchvision
import torch
import timm

from ..classifiers import *
from ..experiments.domain_adaptation import *


def split_attr(
    module_path: str,
    default_module = __name__
) -> Tuple[Optional[str], str]:
    components = module_path.split(".")
    module = default_module
    if len(components) > 1:
        module = ".".join(components[:-1])
    cls = components[-1]
    return sys.modules[module], cls

class ConfigHandleable:
    DEFAULT_MODULE = __name__

    # --- config
    @classmethod
    def save_config(
        cls, obj: Dict, fp: str
    ):
        ext = fp.split(".")[-1]
        if os.path.isfile(fp):
            getattr(cls, f"_save_{ext}")(obj, fp)

    @classmethod
    def _save_yml(cls, obj: Dict, fp: str):
        with open(fp, 'w') as f:
            yaml.dump(obj, f, default_flow_style=False)
    @classmethod
    def _save_json(cls, obj: Dict, fp: str):
        with open(fp, 'w') as f:
            json.dump(obj, f, indent=4)

    @classmethod
    def load_config(
        cls, fp: Union[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        if not isinstance(fp, str):
            return fp
        ext = fp.split(".")[-1]
        if os.path.isfile(fp):
            return getattr(cls, f"_load_{ext}")(fp)
        raise FileNotFoundError(f"{fp} cannot be found or a file.")

    @classmethod
    def _load_yml(cls, fp: str) -> Dict[str, Any]:
        with open(fp, "r") as f:
            data = yaml.safe_load(f)
        return data
    @classmethod
    def _load_json(cls, fp: str) -> Dict[str, Any]:
        with open(fp, "r") as f:
            data = json.load(f)
        return data
    
    # --- transform
    @classmethod
    def parse_transform(
        cls, transform_op: str, 
        transform_kwargs: Optional[Dict[str, Any]]
    ) -> Callable:
        obj_module, transform_op = split_attr(transform_op)
        if transform_kwargs is None:
            return getattr(obj_module, transform_op)()
        for arg in transform_kwargs:
            if isinstance(transform_kwargs[arg], str):
                try:
                    param_module, param_cls = split_attr(transform_kwargs[arg], cls.DEFAULT_MODULE)
                    transform_kwargs[arg] = getattr(param_module, param_cls)
                except:
                    continue
        return getattr(obj_module, transform_op)(**transform_kwargs)
    
    @classmethod
    def parse_transforms(
        cls, config: Dict[str, Any]
    ) -> T.Compose:
        transforms = []
        for transform_op, transform_kwargs in config.items():
            transforms.append(cls.parse_transform(
                transform_op, transform_kwargs))
        return T.Compose(transforms)

    # --- dataset
    @classmethod
    def parse_dataset(
        cls, config: Dict[str, Any], 
        transforms: Optional[Union[T.Compose, Dict[str, Any]]] = None
    ) -> Dataset:
        if isinstance(transforms, Dict):
            transforms = cls.parse_transforms(transforms)
        config["args"]["transforms"] = transforms

        dataset_module, dataset_cls = split_attr(config["type"], cls.DEFAULT_MODULE)
        return getattr(dataset_module, dataset_cls)(**config["args"])

    # --- dataloader
    @classmethod
    def parse_dataloader(
        cls, dataset: Dataset, config: Dict[str, Any]
    ) -> DataLoader:
        return DataLoader(dataset, **config)
    
    # --- model
    @classmethod
    def parse_model(
        cls, config: Dict[str, Any]
    ) -> nn.Module:
        model_module, model_cls = split_attr(config["type"], cls.DEFAULT_MODULE)
        return getattr(model_module, model_cls)(**config["args"])
    
    # --- loss
    @classmethod
    def parse_loss(
        cls, config: Dict[str, Any]
    ) -> Callable:
        config = cls.__normalize_config_args(config)
        loss_module, loss_cls = split_attr(config["type"], cls.DEFAULT_MODULE)
        return getattr(loss_module, loss_cls)(**config["args"])

    # --- optimizer
    @classmethod
    def parse_optimizer(
        cls, config: Dict[str, Any],
        model_params: Any
    ) -> torch.optim.Optimizer:
        config = cls.__normalize_config_args(config)
        optim_module, optim_cls = split_attr(config["type"], cls.DEFAULT_MODULE)
        return getattr(optim_module, optim_cls)(model_params, **config["args"])
    
    # --- scheduler
    @classmethod
    def parse_scheduler(
        cls, config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler.LRScheduler:
        config = cls.__normalize_config_args(config)
        scheduler_module, scheduler_cls = split_attr(config["type"], cls.DEFAULT_MODULE)
        return getattr(scheduler_module, scheduler_cls)(optimizer=optimizer, **config["args"])
    
    @classmethod
    def __normalize_config_args(
        cls, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        config = copy.deepcopy(config)
        if config["args"] is None:
            config["args"] = {}
        return config

    


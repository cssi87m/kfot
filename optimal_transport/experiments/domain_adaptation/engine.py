import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import wandb
import copy
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Any, Union, Optional, Tuple

from .utils import load_features
from .datasets import FeatureDataset
from ...classifiers import ChenXiMLP
from ...adapters._ot import OT
from ...utils.configs import ConfigHandleable


class DomainAdaptationEngine(ConfigHandleable):
    def __init__(self, *args, **kwargs):
        # init wandb
        self.logger = wandb.init(project="kfot", group="domain_adaptation")

    def __call__(
        self,
        model_config: Union[Dict[str, Any], str],
        source_config: Union[Dict[str, Any], str],
        target_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        ckpt_path: str,
    ):
        # TODO: train on source
        # TODO: fine-tune on target data (partly)
        # TODO: adapt source to target
        # TODO: (re) fine-tune on source (adapted) + target (partly)

        self.train(ckpt_path, model_config, engine_config, target_config, source_config)

    def train(
        self,
        ckpt_path: str,
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str],
        **kwargs
    ):
        dataloader, model, (start_epoch, optimizer, scheduler, criterion), device = \
            self.__setup_resources(ckpt_path, model_config, engine_config, *data_configs)
        model.train()

        num_corrects, total_images = 0, 0
        for epoch in (pbar := tqdm(range(start_epoch, engine_config["num_epochs"]), 
                                   total=engine_config["num_epochs"])):
            for _, (features, labels) in enumerate(dataloader):
                outputs = model(features.to(device))
                _, preds = torch.max(outputs, 1)
                loss: torch.Tensor = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                num_corrects += torch.sum(preds == labels)
                total_images += len(features)
                pbar.set_postfix({"accuracy": num_corrects / total_images})
                self.logger.log({"accuracy": num_corrects / total_images})
            scheduler.step()

            if epoch % engine_config["valid_freq"] == 0:
                torch.save(dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    config=model_config,
                    epoch=epoch,
                    metric={"accuracy": num_corrects / total_images}
                ), ckpt_path)

    def evaluate(
        self,
        ckpt_path: str,
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str],
        **kwargs
    ):
        dataloader, model, (start_epoch, optimizer, scheduler, criterion), device = \
            self.__setup_resources(ckpt_path, model_config, engine_config, *data_configs)
        model.eval()

        num_corrects, total_images = 0, 0
        for _, (features, labels) in (pbar := tqdm(enumerate(dataloader), total=len(dataloader))):
            outputs = model(features.to(device))
            _, preds = torch.max(outputs, 1)

            num_corrects += torch.sum(preds == labels)
            total_images += len(features)
            self.logger.log({"accuracy": num_corrects / total_images})
            pbar.set_postfix({"accuracy": num_corrects / total_images})

    def adapt_domain(
        self,
        adapter: OT,
        source_features: Union[str, np.ndarray],
        target_features: Union[str, np.ndarray],
        **kwargs
    ) -> np.ndarray:
        if isinstance(source_features, str):
            source_features = np.array(load_features(source_features))
        if isinstance(target_features, str):
            target_features = np.array(load_features(target_features))

        n, m = source_features.shape[0], target_features.shape[0]
        adapter.fit(source_features, target_features, a=1/n*np.ones(n), b=1/m*np.ones(m), **kwargs)
        return adapter.transport(source_features, target_features)


    def __setup_resources(
        self,
        ckpt_path: str, 
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str],
        **kwargs
    ) -> Tuple[DataLoader, nn.Module, Tuple[int, torch.nn.Module, Optional[torch.optim.Optimizer], 
               Optional[torch.optim.lr_scheduler.LRScheduler]], str]:
        # load configs
        engine_config = self.__validate_config(engine_config)

        # create datasets and dataloaders
        datasets = []
        for data_config in data_configs:
            data_config = self.__validate_config(data_config)
            dataset = FeatureDataset(data_config["feature"], data_config["args"]["annotation_file"])
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        dataloader = self.parse_dataloader(dataset, engine_config["train"]["dataloader"])

        # setup device
        device = model_config["device"] if torch.cuda.is_available() else "cpu"

        # create models
        if model_config["model"]["type"] == "chenxi_mlp":
            model_config["model"]["args"]["input_dims"] = dataset.feature.shape[1]
        model = self.parse_model(model_config["model"]).to(device)

        # create loss and optimizer/scheduler
        optimizer = self.parse_optimizer(engine_config["optimizer"], model.parameters())
        scheduler = self.parse_scheduler(engine_config["scheduler"], optimizer)
        criterion = self.parse_loss(engine_config["loss"])

        # load checkpoint (if had)
        start_epoch, model, optimizer, scheduler = self.load_checkpoint(
            ckpt_path, model, optimizer, scheduler
        )

        return dataloader, model, (start_epoch, optimizer, scheduler, criterion), device

    def __validate_config(
        self, 
        config: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        if isinstance(config, str):
            config = self.load_config(config)
        return config
    
    @classmethod
    def load_checkpoint(
        cls,
        ckpt_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None, 
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None, 
        **kwargs
    ) -> Tuple[int, torch.nn.Module, Optional[torch.optim.Optimizer], 
               Optional[torch.optim.lr_scheduler.LRScheduler]]:
        if not (os.path.isfile(ckpt_path) \
                and ckpt_path.split(".")[-1] == "pth"):
            return 0, model, optimizer, scheduler
        
        ckpt = torch.load(ckpt_path, map_location="cpu")
        try:
            model = model.load_state_dict(ckpt["model"])
        except:
            model = cls.parse_model(ckpt["config"])
            model = model.load_state_dict(ckpt["model"])
        if optimizer:
            optimizer = optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler:
            scheduler = scheduler.load_state_dict(ckpt["scheduler"])

        return ckpt["epoch"], model, optimizer, scheduler



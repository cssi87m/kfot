import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import os
import wandb
import copy
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, Any, Union, Optional, Tuple, List

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
        adapter: OT,
        model_config: Union[Dict[str, Any], str],
        source_config: Union[Dict[str, Any], str],
        target_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        ckpt_path: str,
    ):
        # DONE: train on source
        # DONE: adapt source to target
        # DONE: (re) fine-tune on source (adapted) + target (partly)

        # train on source data
        self.train(
            ckpt_path, "source", model_config, engine_config, 
            source_config)
        self.evaluate(
            ckpt_path, model_config, engine_config, 
            source_config)
        
        # split target data
        lb_target_config = copy.deepcopy(target_config)
        lb_target_config["sample_size"] = (0, engine_config["task"]["target"]["sample_size"])
        ulb_target_config = copy.deepcopy(target_config)
        ulb_target_config["sample_size"] = (engine_config["task"]["target"]["sample_size"], 1)

        self.evaluate(
            ckpt_path, model_config, engine_config, 
            target_config)
        
        # adapt source to target data
        lb_target_config = self.__validate_config(lb_target_config)
        lb_target_dataset = FeatureDataset(lb_target_config["feature"], **lb_target_config["args"])
        
        K = self.__setup_keypoints(
            np.array(lb_target_dataset.features), 
            np.array(lb_target_dataset.targets),
            n_shot=engine_config["task"]["n_shot"]
        )

        adapted_source_dataset = self.adapt_domain(adapter, 
            source_config["feature"], target_config["feature"], K=K)
        
        # fine-tune on the adapted and target data
        self.train(
            ckpt_path, "target", model_config, engine_config, 
            adapted_source_dataset, lb_target_dataset)
        self.evaluate(
            ckpt_path, "target", model_config, engine_config, 
            target_config)

    def train(
        self,
        ckpt_path: str, mode: str,
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str, FeatureDataset],
        **kwargs
    ):
        dataloader, model, (start_epoch, optimizer, scheduler, criterion), device = \
            self.__setup_resources(ckpt_path, model_config, engine_config, *data_configs)
        model.train()

        num_corrects, total_images = 0, 0
        for epoch in (pbar := tqdm(range(start_epoch, engine_config[mode]["num_epochs"]), 
                                   total=engine_config[mode]["num_epochs"])):
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

            if epoch % engine_config["save_freq"] == 0:
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
        ckpt_path: str, mode: str,
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str, FeatureDataset],
        **kwargs
    ):
        dataloader, model, _, device = \
            self.__setup_resources(ckpt_path, mode, model_config, engine_config, *data_configs)
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
        source_config: Union[Dict[str, Any], str],
        target_config: Union[Dict[str, Any], str],
        **kwargs
    ) -> FeatureDataset:
        source_config = self.__validate_config(source_config)
        target_config = self.__validate_config(target_config)

        source_dataset = FeatureDataset(source_config["feature"], **source_config["args"])
        target_dataset = FeatureDataset(target_config["feature"], **target_config["args"])
        source_features = np.array(source_dataset.features)
        target_features = np.array(target_dataset.features)

        n, m = source_features.shape[0], target_features.shape[0]
        adapter.fit(source_features, target_features, 
                    a=1/n*np.ones(n), b=1/m*np.ones(m), **kwargs)
        source_dataset.features = adapter.transport(source_features, target_features)

        return source_dataset
    
    def __setup_keypoints(
        self, x: np.ndarray, y: np.ndarray = None,
        n_shot: int = 1
    ) -> List[int]:
        labels = np.unique(y)
        selected_inds = []
        for label in labels:
            cls_indices = np.where(y == label)[0]
            distance = self.dist_fn(x[cls_indices], np.mean(x[cls_indices], axis=0)[None, :]).squeeze()
            selected_inds.extend(cls_indices[np.argsort(distance)[:n_shot]])
        return selected_inds

    def __setup_resources(
        self,
        ckpt_path: str, mode: str,
        model_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        *data_configs: Union[Dict[str, Any], str, FeatureDataset],
        **kwargs
    ) -> Tuple[DataLoader, nn.Module, Tuple[int, torch.nn.Module, Optional[torch.optim.Optimizer], 
               Optional[torch.optim.lr_scheduler.LRScheduler]], str]:
        # load configs
        engine_config = self.__validate_config(engine_config)

        # create datasets and dataloaders
        datasets = []
        for data_config in data_configs:
            if not isinstance(data_config, FeatureDataset): 
                data_config = self.__validate_config(data_config)
                dataset = FeatureDataset(data_config["feature"], **data_config["args"])
            datasets.append(dataset)
        dataset = ConcatDataset(datasets)
        dataloader = self.parse_dataloader(dataset, engine_config["dataloader"])

        # setup device
        device = model_config["device"] if torch.cuda.is_available() else "cpu"

        # create models
        if model_config["model"]["type"] == "chenxi_mlp":
            model_config["model"]["args"]["input_dims"] = dataset.feature.shape[1]
        model = self.parse_model(model_config["model"]).to(device)

        # create loss and optimizer/scheduler
        optimizer = self.parse_optimizer(engine_config[mode]["optimizer"], model.parameters())
        scheduler = self.parse_scheduler(engine_config[mode]["scheduler"], optimizer)
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
            try:
                optimizer = optimizer.load_state_dict(ckpt["optimizer"])
            except:
                pass
        if scheduler:
            try:
                scheduler = scheduler.load_state_dict(ckpt["scheduler"])
            except:
                pass

        return ckpt["epoch"], model, optimizer, scheduler



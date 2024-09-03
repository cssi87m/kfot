import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, Any, Union, Optional

from .utils import load_features
from .datasets import OfficeFeature
from ...classifiers import ChenXiMLP
from ...adapters._ot import OT
from ...utils.configs import ConfigHandleable


class DomainAdaptationEngine(ConfigHandleable):
    def __init__(self):
        super().__init__()

    def __call__(self):
        # TODO: train on source
        # TODO: fine-tune on target data (partly)
        # TODO: adapt source to target
        # TODO: (re) fine-tune on source (adapted) + target (partly)

        pass

    def train(
        self,
        model_config: Union[Dict[str, Any], str],
        source_config: Union[Dict[str, Any], str],
        target_config: Union[Dict[str, Any], str],
        engine_config: Union[Dict[str, Any], str],
        **kwargs
    ):
        # load configs
        source_config = self.__validate_config(source_config)
        target_config = self.__validate_config(target_config)
        engine_config = self.__validate_config(engine_config)

        # create datasets and dataloaders
        source_dataset = OfficeFeature(source_config["feature"], source_config["args"]["annotation_file"])
        target_dataset = OfficeFeature(target_config["feature"], target_config["args"]["annotation_file"]) 
        source_dataloader = self.parse_dataloader(source_dataset, source_config["dataloader"])
        target_dataloader = self.parse_dataloader(target_dataset, target_config["dataloader"])

        assert source_dataset.feature.shape[1] == target_dataset.feature.shape[1], "Two feature sets must share the same dimensions."

        # setup device
        device = model_config["device"] if torch.cuda.is_available() else "cpu"

        # create models
        if model_config["model"]["type"] == "chenxi_mlp":
            model_config["model"]["args"]["input_dims"] = source_dataset.feature.shape[1]
        model = self.parse_model(model_config["model"]).to(device)
        model.train()

        # create loss and optimizer
        optimizer = self.parse_optimizer(engine_config["optimizer"], model.parameters())
        criterion = self.parse_loss(engine_config["loss"])

        for epoch in (pbar := tqdm(range(engine_config["num_epochs"]), total=engine_config["num_epochs"])):
            num_corrects, total_images = 0, 0
            for _, (features, labels) in source_dataloader:
                outputs = model(features.to(device))
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                num_corrects += torch.sum(preds == labels)
                total_images += len(features)
                pbar.set_postfix({"accuracy": num_corrects / total_images})

            if epoch % engine_config["valid_freq"] == 0:
                # evaluate
                pass

    def evaluate(self):
        pass

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


    def __validate_config(
        self, 
        config: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        if isinstance(config, str):
            config = self.load_config(config)
        return config

import torch
from tqdm import tqdm
import pickle as pkl
from typing import Dict, Any, Union, Optional

from ...utils.configs import ConfigHandleable
from .utils import save_features, load_features


class FeatureExtractor(ConfigHandleable):
    def __init__(
        self, 
        model_config: Union[Dict[str, Any], str],
        **kwargs
    ):  
        super().__init__()
        self.model_config = self.load_config(model_config)
        self.device = self.model_config["device"] if torch.cuda.is_available() else "cpu"
        
        self.model = self.parse_model(self.model_config["model"])
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def __call__(
        self, 
        data_config: Union[Dict[str, Any], str],
        fp: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        data_config = self.load_config(data_config)
        dataset = self.parse_dataset(
            data_config["dataset"], 
            transforms=self.model_config["transforms"]
        )
        dataloader = self.parse_dataloader(
            dataset, data_config["dataloader"])

        all_feats = []
        with torch.no_grad():
            for _, (imgs, _) in tqdm(enumerate(dataloader), total=len(dataloader)):
                feats = self.model(imgs.to(self.device))
                all_feats.append(feats.to("cpu"))
        all_feats = torch.concat(all_feats)

        if fp is None:
            try:
                fp = data_config["feature"]
            except KeyError:
                return feats
        self.save_features(fp, feats)
        return feats

    @classmethod 
    def save_features(
        cls, fp: str,feats: torch.Tensor
    ):
        save_features(fp, feats)

    @classmethod
    def load_features(
        cls, fp: str,
    ) -> torch.Tensor:
        load_features(fp)
import torch
from tqdm import tqdm
import pickle as pkl
from typing import Dict, Any, Union, Optional

from ...utils.configs import ConfigHandleable


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
        dataset = self.parse_dataset(
            data_config["dataset"], 
            transforms=self.model_config["transforms"]
        )
        dataloader = self.parse_dataloader(
            dataset, data_config["dataloader"])

        all_feats = []
        for _, (imgs, _) in tqdm(enumerate(dataloader)):
            feats = self.model(imgs.to(self.device))
            all_feats.append(feats.to("cpu"))
        all_feats = torch.concat(all_feats)

        if fp is None:
            try:
                fp = data_config["features"]
            except KeyError:
                return feats
        self.save_features(fp, feats)
        return feats

    @classmethod 
    def save_features(
        cls, fp: str,
        feats: torch.Tensor
    ):
        with open(fp, "wb") as f:
            pkl.dump(feats, f)

    @classmethod
    def load_features(
        cls, fp: str,
    ) -> torch.Tensor:
        with open(fp, 'rb') as f:
            feats = pkl.load(f)
        return feats
import torch
import numpy as np
from typing import Dict, Any, Union, Optional

from .utils import load_features, save_features
from .datasets import OfficeFeature
from ...models._ot import OT
from ...utils.configs import ConfigHandleable


class DomainAdaptationEngine(ConfigHandleable):
    def __init__(self):
        super().__init__()
    
    def train(self):
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



from typing import Dict, Any
import timm
import sys

from .datasets import *


def extract_features(
    config: Dict[str, Any]
):
    model = timm.create_model(
        config["model"]["id"], 
        pretrained=config["model"]["pretrained"], 
        num_classes=0
    )
    dataset = getattr(
        sys.modules[__name__], 
        config["dataset"]["id"], None)
    
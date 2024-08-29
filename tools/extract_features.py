import sys
from os.path import dirname
import os
import argparse

ROOT = dirname(dirname(sys.modules[__name__].__file__))
sys.path.append(ROOT)

from optimal_transport.experiments.domain_adaptation \
    import FeatureExtractor
from optimal_transport.utils.configs \
    import ConfigHandleable


def main(
    model_cp: str,
    data_cp: str,
    feat_fp: str,
):
    fn = FeatureExtractor(model_cp)
    fn(data_cp, feat_fp)

    data_config = ConfigHandleable.load_config(data_cp)
    data_config["feature"] = feat_fp
    ConfigHandleable.save_config(data_config, data_cp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_config', type=str, 
                        default=os.path.join(ROOT, "configs/classifiers/resnet50_pooled.yml"))
    parser.add_argument('--data_config', type=str, 
                        default=os.path.join(ROOT, "configs/datasets/office31/amazon.yml"))
    parser.add_argument('--feature_file', type=str, 
                        default=os.path.join(ROOT, "datasets/checkpoints/resnet50_amazon.pkl"))
    
    args = parser.parse_args()

    main(args.model_config, args.data_config, args.feature_file)
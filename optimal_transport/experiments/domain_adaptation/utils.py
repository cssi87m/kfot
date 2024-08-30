import torch
import pickle as pkl


def save_features(
    fp: str,
    feats: torch.Tensor
):
    with open(fp, "wb") as f:
        pkl.dump(feats, f)


def load_features(
    fp: str,
) -> torch.Tensor:
    with open(fp, 'rb') as f:
        feats = pkl.load(f)
    return feats
import numpy as np


def accuracy(
    preds: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    return np.sum(preds == labels) / labels.shape[0]

def error_correction(
    preds: np.ndarray, preds_: np.ndarray, 
    labels: np.ndarray
) -> np.ndarray:
    acc, acc_ = accuracy(preds, labels), accuracy(preds_, labels)
    return (acc_ - acc) / (1 - acc)
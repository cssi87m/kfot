import numpy as np 


def l2(
    x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    return np.expand_dims((x**2).sum(axis=1),1) + np.expand_dims((y**2).sum(axis=1),0) - 2 * x@y.T

def kl_div(
    x: np.ndarray, y: np.ndarray, 
    eps: float = 1e-10
) -> np.ndarray:
    return np.sum(x * np.log(x + eps) - x * np.log(y + eps), axis=-1)

def js_div(
    x: np.ndarray, y: np.ndarray, 
    eps: float = 1e-10
) -> np.ndarray:
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=0)
    return 0.5 * (kl_div(x, (x + y) / 2, eps) + kl_div(y, (x + y) / 2, eps))
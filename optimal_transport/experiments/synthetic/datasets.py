import numpy as np
from typing import List, Tuple, Dict, Optional


def gauss_mixture(
    samples_per_cluster: int = 20,
    n_clusters: int = 3,
    n_dims: int = 2,
    centroids: Optional[Dict[str, List[np.ndarray]]] = {
        "source": [np.array([[-1, -1]]),np.array([[-3, 2]]),np.array([[-2, 3]])],
        "target": [np.array([[0, 1]]),np.array([[-0.5, 0.5]]),np.array([[-1, 2]])]
    },
    covariances: Optional[Dict[str, List[np.ndarray]]] = {
        "source": [0.05 * np.array([[1, 0], [0, 1]])] * 3,
        "target": [0.05 * np.array([[1, 0], [0, 1]])] * 3
    },
    random_seed: int = 3
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    
    np.random.seed(random_seed)
    source, source_kp_idxs = [], []
    target, target_kp_idxs = [], []
    
    for i in range(n_clusters):
        source_kp_idxs.append(i * samples_per_cluster)
        source.append(np.concatenate(
            (centroids["source"][i],
            np.random.multivariate_normal(
                np.array([0 for _ in range(n_dims)]), 
                cov=covariances["source"][i], 
                size=samples_per_cluster-1) + centroids["source"][i]), axis=0)
        )
        target_kp_idxs.append(i * samples_per_cluster)
        target.append(np.concatenate(
            (centroids["target"][i],
            np.random.multivariate_normal(
                np.array([0 for _ in range(n_dims)]), 
                cov=covariances["target"][i], 
                size=samples_per_cluster-1) + centroids["target"][i]), axis=0)
        )
    
    return source, target, \
           source_kp_idxs, target_kp_idxs
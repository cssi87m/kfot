from .._ot import OT

from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
import ot


class FOT(OT):
    def __init__(
        self,
        div_term: float = 1e-10,
        distance: Callable = ot.dist,
        solver_kwargs: Dict[str, Any] = {
            "r": 3, "reg": 0.001, "stopThr": 1e-7,
            "numItermax": 10000, "method": "sinkhorn_log"
        }
    ):
        super().__init__(distance, div_term)
        self.solver_kwargs = solver_kwargs

        self.Pi_: List[Optional[np.ndarray]] = [None, None]
        self.z_: Optional[np.ndarray] = None
    
    def fit(
        self, 
        xs: np.ndarray, xt: np.ndarray, 
        a: Optional[np.ndarray], b: Optional[np.ndarray],
        **kwargs,
    ) -> "FOT":
        z0, _ = self._init_anchors(xs, self.solver_kwargs["r"])
        self.Pi_[0], self.Pi_[1], self.z_ = ot.factored.factored_optimal_transport(
            xs, xt, a, b, X0=z0, **self.solver_kwargs)
        
        self.P_ = np.dot(
            self.Pi_[0] / (self.Pi_[0].T.dot(np.ones([xs.shape[0], 1]))).T, 
            self.Pi_[1])
        return self

    def transport(
        self,
        xs: np.ndarray, xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n = xs.shape[0]
        m = xt.shape[0]
        assert (self.Pi_[0] is not None) and (self.Pi_[1] is not None), "Should run fit() before mapping"
        
        Cx = self.Pi_[0].T.dot(xs) / (self.Pi_[0].T.dot(np.ones((n, 1))) + self.div_term)
        Cy = self.Pi_[1].dot(xt) / (self.Pi_[1].dot(np.ones((m, 1))) + self.div_term)
        return xs + np.dot(self.Pi_[0] / np.sum(self.Pi_[0], axis=1).reshape([n, 1]), Cy - Cx)


    def _init_anchors(
        self, 
        x: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        model = KMeans(n_clusters=n_clusters)
        model.fit(x)
        Z = model.cluster_centers_
        h = np.ones(n_clusters) / (n_clusters)
        return Z, h
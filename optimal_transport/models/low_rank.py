from ._ot import OT

from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
import ot


class LrSinkhornOT(OT):
    def __init__(
        self,
        div_term: float = 1e-10,
        distance: Callable = ot.dist,
        solver_kwargs: Dict[str, Any] = dict(
            reg=0, rank=3, alpha=1e-10, rescale_cost=False, init="random",
            reg_init=0.1, seed_init=49, gamma_init="rescale", numItermax=2000,
            stopThr=1e-7, warn=True
        )
    ):
        super().__init__(distance, div_term)
        self.div_term = div_term
        self.solver_kwargs = solver_kwargs

        self.Pi_: List[Optional[np.ndarray]] = [None, None]
        self.z_: Optional[np.ndarray] = None
    
    def fit(
        self, 
        xs: np.ndarray, xt: np.ndarray, 
        a: Optional[np.ndarray], b: Optional[np.ndarray],
        **kwargs,
    ) -> "LrSinkhornOT":
        Q, R, g = ot.lowrank.lowrank_sinkhorn(
            xs, xt, a, b, **self.solver_kwargs
        )
        self.z_ = g
        self.P_ = Q.dot(np.diag(1/g)).dot(R.T)
        return self
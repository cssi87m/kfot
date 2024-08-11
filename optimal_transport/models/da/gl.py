from .._ot import OT

from typing import Optional, List, Tuple, Callable, Dict, Any
import numpy as np
from sklearn.cluster import KMeans
import ot


class GlOTDA(OT):
    def __init__(
        self,
        div_term: float = 1e-10,
        distance: Callable = ot.dist,
        solver_kwargs: Dict[str, Any] = dict(
            reg=0.1, eta=0.1, numItermax=10, numInnerItermax=200, 
            stopInnerThr=1e-09, eps=1e-12
        )
    ):
        super().__init__(distance, div_term)
        self.solver_kwargs = solver_kwargs

        self.z_: Optional[np.ndarray] = None

    def fit(
        self,
        xs: np.ndarray, xt: np.ndarray,
        ys: np.ndarray,
        a: Optional[np.ndarray], b: Optional[np.ndarray],
        **kwargs,
    ) -> "GlOTDA":
        C = self.dist_fn(xs, xt)
        C = C / (C.max() + self.div_term)

        self.P_ = ot.da.sinkhorn_l1l2_gl(
            a, ys, b, C, **self.solver_kwargs
        )

        return self

from ._ot import OT

import numpy as np
from typing import Optional, Callable, Dict, Any
import ot


class EntropicOT(OT):
    def __init__(
        self,
        distance: Callable = ot.dist,
        div_term: float = 1e-10,
        solver_kwargs: Dict[str, Any] = {
            "reg": 0.01, "method": "sinkhorn_log", 
            "numItermax": 10000, "stopThr": 1e-10, "warmstart": None
        },
    ):
        super().__init__(distance, div_term)
        self.solver_kwargs = solver_kwargs
    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "EntropicOT":
        C = self.dist_fn(xs, xt)
        C = C / (C.max() + self.div_term)
        
        self.P_ = ot.sinkhorn(a, b, C, **self.solver_kwargs)
        return self


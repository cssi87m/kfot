from ._ot import OT

import numpy as np
from typing import Optional, Callable, Dict, Any
import ot


class EMD(OT):
    def __init__(
        self,
        distance: Callable = ot.dist,
        div_term : int = 1e-10,
        solver_kwargs: Dict[str, Any] = {"maxItermax": 10000}
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
    ) -> "EMD":
        C = self.dist_fn(xs, xt)
        C = C / (C.max() + self.div_term)
        
        self.P_ = ot.emd(a, b, C, **self.solver_kwargs)
        return self


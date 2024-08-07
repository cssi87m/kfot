import numpy as np
from typing import Callable, Optional, Any


class OT:
    def __init__(
        self,
        distance: Optional[Callable],
        div_term: float = 1e-10,
        **kwargs
    ):
        self.dist_fn = distance
        self.div_term = div_term
        self.P_: Optional[np.ndarray] = None
    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "OT":
        pass

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        assert self.P_ is not None, "Should run fit() before mapping"
        return self.P_.dot(xt) / (self.P_.dot(np.ones((xt.shape[0], 1))) + self.div_term)

    def objective(self, **kwargs) -> np.ndarray:
        pass


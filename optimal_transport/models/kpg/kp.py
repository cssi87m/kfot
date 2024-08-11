from .._ot import OT
from ...functional.distances import js_div
from ...functional.ops import softmax

from typing import Optional, List, Tuple, Callable
import numpy as np
import ot


class KPGRLKP(OT):
    def __init__(
        self,
        distance: Callable = ot.dist,
        div_term: float = 1e-10,
        similarity: Callable = js_div,
        sinkhorn_reg: float = 0.0001, 
        temperature: float = 0.1,  
        guide_mixing: float = 0.5,
        stop_thr: float = 1e-5, 
        max_iters: int = 10000,
    ):
        super().__init__(distance, div_term)
        self.sim_fn = similarity

        self.eps = sinkhorn_reg
        self.rho = temperature
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.alpha = 1 - guide_mixing

    def fit(
        self,
        xs: np.ndarray, xt: np.ndarray, 
        a: Optional[np.ndarray], b: Optional[np.ndarray], 
        K: List[Tuple], **kwargs,
    ) -> "KPGRLKP":
        I, J = self._init_keypoint_inds(K)
        M = self._guide_mask(xs, xt, I, J)
        
        C = self.dist_fn(xs, xt)
        C = C / (C.max() + self.div_term)
        G = self.alpha * C + (1 - self.alpha) * self._guide_matrix(xs, xt, I, J)

        self.P_ = self._sinkhorn_log_domain(a, b, G, M)
        return self


    def _init_keypoint_inds(
        self,
        K: List[Tuple]
    ) -> Tuple[np.ndarray]:
        I = np.array([pair[0] for pair in K])
        J = np.array([pair[1] for pair in K])
        return I, J
    
    def _sinkhorn_log_domain(
        self,
        p: np.ndarray, q: np.ndarray,
        C: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        C = C / (C.max() + self.div_term)

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / self.eps
            if mask is not None:
                M[mask==0] = -1e6
            return M

        def lse(A):
            "log-sum-exp"
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A-max_A).sum(1, keepdims=True) + self.div_term) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        for i in range(self.max_iters):
            u1 = u  # useful to check the update
            u = self.eps * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)
            if err < self.stop_thr:
                break

        U, V = u, v
        P = np.exp(M(U, V))  # P = diag(a) * K * diag(b)
        return P

    def _guide_mask(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        mask = np.ones((xs.shape[0], xt.shape[0]))
        mask[I, :] = 0
        mask[:, J] = 0
        mask[I, J] = 1
        return mask

    def _guide_matrix(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray,
    ) -> np.ndarray:
        Cs = self.dist_fn(xs, xs)
        Ct = self.dist_fn(xt, xt)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = Ct / (Ct.max() + self.div_term)

        Cs_kp = Cs[:, I]
        Ct_kp = Ct[:, J]
        R1 = softmax(-2 * Cs_kp / self.rho)
        R2 = softmax(-2 * Ct_kp / self.rho)
        G = self.sim_fn(R1, R2, eps=self.div_term)
        return G 
from ._ot import OT

import numpy as np
from typing import Optional, Callable, Dict, Any
import ot
from functools import partial


def ratio(P, Kx, Ky):
    '''
    compute the ratio berween joint and marginal densities
    Parameters
    ----------
    P : transportation plan
    Kx: source kernel matrix
    Ky: target kernel matrix

    Returns
    ----------
    ratio matrix for (x_i, y_j)
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = np.outer(f_x, f_y)
    constC = np.zeros((len(Kx), len(Ky)))
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    return f_xy / f_x_f_y

def compute_kernel(Cx, Cy, h):
    '''
    compute Gaussian kernel matrices
    Parameters
    ----------
    Cx: source pairwise distance matrix
    Cy: target pairwise distance matrix
    h : bandwidth
    Returns
    ----------
    Kx: source kernel
    Ky: targer kernel
    '''
    std1 = np.sqrt((Cx**2).mean() / 2)
    std2 = np.sqrt((Cy**2).mean() / 2)
    h1 = h * std1
    h2 = h * std2
    # Gaussian kernel (without normalization)
    Kx = np.exp(-(Cx / h1)**2 / 2)
    Ky = np.exp(-(Cy / h2)**2 / 2)
    return Kx, Ky

def migrad(P, Kx, Ky):
    '''
    compute the gradient w.r.t. KDE mutual information
    Parameters
    ----------
    P : transportation plan
    Ks: source kernel matrix
    Kt: target kernel matrix

    Returns
    ----------
    negative gradient w.r.t. MI
    '''
    f_x = Kx.sum(1) / Kx.shape[1]
    f_y = Ky.sum(1) / Ky.shape[1]
    f_x_f_y = np.outer(f_x, f_y)
    constC = np.zeros((len(Kx), len(Ky)))
    # Negative sign in ot.gromov.tensor_product
    f_xy = -ot.gromov.tensor_product(constC, Kx, Ky, P)
    P_f_xy = P / f_xy
    P_grad = -ot.gromov.tensor_product(constC, Kx, Ky, P_f_xy)
    P_grad = np.log(f_xy / f_x_f_y) + P_grad
    return -P_grad

def projection(P, X):
    '''
    compute the projection based on similarity matrix
    Parameters
    ----------
    P : transportation plan or similarity matrix
    X : target data

    Returns
    ----------
    projected source data
    '''
    weights = np.sum(P, axis = 1)
    X_proj = np.matmul(P, X) / weights[:, None]
    return X_proj


class InfoOT(OT):
    def __init__(
        self,
        distance: Callable = partial(ot.dist, metric="euclidean"),
        div_term: float = 1e-10,
        sinkhorn_reg: float = 0.05,
        num_iters: int = 100,
        bandwidth: float = 0.5
    ):
        super().__init__(distance, div_term)
        self.sinkhorn_reg = sinkhorn_reg
        self.num_iters = num_iters
        self.bandwidth = bandwidth

    def fit(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray]
    ) -> "InfoOT":
        '''
        Solve projected gradient descent via sinkhorn iteration
        '''
        Cs = self.dist_fn(xs, xs)
        Ct = self.dist_fn(xt, xt)
        Ks, Kt = compute_kernel(Cs, Ct, self.bandwidth)

        p = np.zeros(len(xs)) + 1. / len(xs)
        q = np.zeros(len(xt)) + 1. / len(xt)
        P = np.outer(p, q)
        for i in range(self.num_iters):
            grad_P = migrad(P, Ks, Kt)
            P = ot.bregman.sinkhorn(p, q, grad_P, reg=self.sinkhorn_reg)
        
        self.P_ = P
        return self
    
    def transport(
        self,
        xs: np.ndarray, xt: np.ndarray,
        method: str = "barycentric",
    ) -> np.ndarray:
        Cs = self.dist_fn(xs, xs)
        Ct = self.dist_fn(xt, xt)

        if method not in ['conditional', 'barycentric']:
            raise Exception('Only suppot conditional or barycebtric projection')

        P = self.P_
        if method == 'conditional':
            Ks, Kt = compute_kernel(Cs, Ct, self.bandwidth)
            P = ratio(self.P_, Ks, Kt)
        return projection(P, xt)
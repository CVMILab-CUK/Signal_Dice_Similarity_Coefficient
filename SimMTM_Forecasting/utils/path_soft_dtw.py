"""Soft-DTW path utilities for DILATE loss.

This file ports the differentiable soft-alignment path computation from the
original DILATE implementation:

    Le Guen, V., & Thome, N. (2019). "Shape and Time Distortion Loss for Training
    Deep Time Series Forecasting Models." NeurIPS 2019.
    https://github.com/vincent-leguen/DILATE/blob/master/loss/path_soft_dtw.py

The functions ``my_max``, ``my_min``, ``my_max_hessian_product``,
``my_min_hessian_product``, ``dtw_grad`` and ``dtw_hessian_prod``, as well as the
``PathDTWBatch`` autograd Function, are adapted from that reference. Numerical
behavior is preserved exactly; only docstrings, naming hygiene, and dtype
handling were updated. The reference repository carries no explicit license file
as of 2026-05-19; this port is performed for academic reproduction (DILATE
baseline comparison in SDSC paper for AAAI 2027).

Differences from the original:
- ``dtw_grad`` uses ``np.float64`` throughout for numerical stability of the
  log-sum-exp soft-min near ``gamma -> 0``.
- ``PathDTWBatch`` constructs intermediate tensors with the same dtype as the
  input ``D`` (was hard-coded ``FloatTensor`` upstream).
"""

import numpy as np
import torch
from torch.autograd import Function
from numba import jit


# --------------------------------------------------------------------- soft-min / soft-max
# Use log-sum-exp form to avoid overflow/underflow in exp(-x/gamma) for small gamma.


@jit(nopython=True)
def my_max(x, gamma):
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / gamma)
    z = np.sum(exp_x)
    return gamma * np.log(z) + max_x, exp_x / z


@jit(nopython=True)
def my_min(x, gamma):
    neg_max, weights = my_max(-x, gamma)
    return -neg_max, weights


@jit(nopython=True)
def my_max_hessian_product(p, z, gamma):
    return (p * z - p * np.sum(p * z)) / gamma


@jit(nopython=True)
def my_min_hessian_product(p, z, gamma):
    return -my_max_hessian_product(p, z, gamma)


# --------------------------------------------------------------------- forward / backward path passes


@jit(nopython=True)
def dtw_grad(theta, gamma):
    """Compute the soft-DTW value, the soft-alignment matrix E, and pointers Q.

    Args:
        theta: (m, n) pairwise cost matrix between two sequences.
        gamma: smoothing parameter of the soft-min.

    Returns:
        v_mn:    scalar soft-DTW(theta, gamma).
        e:       (m, n) soft-alignment path matrix; sum(e) is the expected path length.
        Q:       (m+2, n+2, 3) soft-min weights used by the Hessian product.
        E_full:  (m+2, n+2) padded e (kept for the Hessian backward call).
    """
    m, n = theta.shape
    V = np.zeros((m + 1, n + 1))
    V[:, 0] = 1e10
    V[0, :] = 1e10
    V[0, 0] = 0.0

    Q = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            v, Q[i, j] = my_min(
                np.array([V[i, j - 1], V[i - 1, j - 1], V[i - 1, j]]), gamma
            )
            V[i, j] = theta[i - 1, j - 1] + v

    E = np.zeros((m + 2, n + 2))
    E[m + 1, n + 1] = 1.0
    Q[m + 1, n + 1] = 1.0

    for i in range(m, 0, -1):
        for j in range(n, 0, -1):
            E[i, j] = (
                Q[i, j + 1, 0] * E[i, j + 1]
                + Q[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j, 2] * E[i + 1, j]
            )

    return V[m, n], E[1:m + 1, 1:n + 1], Q, E


@jit(nopython=True)
def dtw_hessian_prod(theta, Z, Q, E, gamma):
    """Hessian-vector product for soft-DTW gradient. Required by ``PathDTWBatch.backward``."""
    m, n = Z.shape

    V_dot = np.zeros((m + 1, n + 1))
    Q_dot = np.zeros((m + 2, n + 2, 3))

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            V_dot[i, j] = (
                Z[i - 1, j - 1]
                + Q[i, j, 0] * V_dot[i, j - 1]
                + Q[i, j, 1] * V_dot[i - 1, j - 1]
                + Q[i, j, 2] * V_dot[i - 1, j]
            )
            v = np.array([V_dot[i, j - 1], V_dot[i - 1, j - 1], V_dot[i - 1, j]])
            Q_dot[i, j] = my_min_hessian_product(Q[i, j], v, gamma)

    E_dot = np.zeros((m + 2, n + 2))
    for j in range(n, 0, -1):
        for i in range(m, 0, -1):
            E_dot[i, j] = (
                Q_dot[i, j + 1, 0] * E[i, j + 1]
                + Q[i, j + 1, 0] * E_dot[i, j + 1]
                + Q_dot[i + 1, j + 1, 1] * E[i + 1, j + 1]
                + Q[i + 1, j + 1, 1] * E_dot[i + 1, j + 1]
                + Q_dot[i + 1, j, 2] * E[i + 1, j]
                + Q[i + 1, j, 2] * E_dot[i + 1, j]
            )

    return V_dot[m, n], E_dot[1:m + 1, 1:n + 1]


# --------------------------------------------------------------------- batched autograd wrapper


class PathDTWBatch(Function):
    """Differentiable soft-alignment path for a batch of cost matrices.

    Inputs:
        D:     (B, N, N) pairwise distance matrix per batch element.
        gamma: scalar soft-min smoothing parameter.

    Output:
        (N, N) soft-alignment path averaged over the batch. Used by DILATE as
        the soft transport plan against the Omega temporal-penalty matrix.
    """

    @staticmethod
    def forward(ctx, D, gamma):
        batch_size, N, _ = D.shape
        device = D.device
        dtype = D.dtype
        D_cpu = D.detach().cpu().numpy().astype(np.float64)
        gamma_t = torch.tensor([gamma], dtype=dtype, device=device)

        grad_all = torch.zeros((batch_size, N, N), dtype=dtype, device=device)
        Q_all = torch.zeros((batch_size, N + 2, N + 2, 3), dtype=dtype, device=device)
        E_all = torch.zeros((batch_size, N + 2, N + 2), dtype=dtype, device=device)

        for k in range(batch_size):
            _, grad_k, Q_k, E_k = dtw_grad(D_cpu[k], float(gamma))
            grad_all[k] = torch.as_tensor(grad_k, dtype=dtype, device=device)
            Q_all[k] = torch.as_tensor(Q_k, dtype=dtype, device=device)
            E_all[k] = torch.as_tensor(E_k, dtype=dtype, device=device)

        ctx.save_for_backward(grad_all, D, Q_all, E_all, gamma_t)
        return torch.mean(grad_all, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        device = grad_output.device
        grad_all, D, Q_all, E_all, gamma_t = ctx.saved_tensors
        dtype = D.dtype
        D_cpu = D.detach().cpu().numpy().astype(np.float64)
        Q_cpu = Q_all.detach().cpu().numpy().astype(np.float64)
        E_cpu = E_all.detach().cpu().numpy().astype(np.float64)
        gamma_val = float(gamma_t.detach().cpu().numpy()[0])
        Z = grad_output.detach().cpu().numpy().astype(np.float64)

        batch_size, N, _ = D_cpu.shape
        hessian = torch.zeros((batch_size, N, N), dtype=dtype, device=device)
        for k in range(batch_size):
            _, hess_k = dtw_hessian_prod(D_cpu[k], Z, Q_cpu[k], E_cpu[k], gamma_val)
            hessian[k] = torch.as_tensor(hess_k, dtype=dtype, device=device)

        return hessian, None


def pairwise_distances(x, y=None):
    """L2 pairwise distance matrix. Convenience helper for DILATE call sites.

    If y is None, computes pairwise distances within x.
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, min=0.0)

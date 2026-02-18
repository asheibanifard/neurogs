#!/usr/bin/env python3
"""
Gaussian Mixture Field (3D) — Refined Training Script
=====================================================

Represents volumetric microscopy data as a mixture of 3D anisotropic Gaussians:

    V(x) = Σ_k  a_k · exp( -½ (x-μ_k)ᵀ Σ_k⁻¹ (x-μ_k) )

with  Σ = R diag(s²) Rᵀ   (rotation quaternion + per-axis log-scales).

Key features
------------
* **Charbonnier reconstruction loss** — robust L1-like, better for thin structures
  than MSE which is dominated by background voxels.
* **Finite-difference gradient supervision** — samples forward-neighbors along each
  axis and penalises mismatch in local intensity gradients.  Critical for preserving
  neurite edges and branch points.
* **Scale regularisation + hard clamping** — prevents Gaussians from inflating into
  large blobs that blur fine detail.
* **Memory-safe K-chunked forward** — avoids materialising full (N, K, 3) tensors;
  memory scales as O(N · chunk) instead of O(N · K).
* **Optional soft-MIP supervision** — LogSumExp approximation to z-axis maximum-
  intensity projection, with annealed temperature τ.
* **Adaptive densification / pruning** — clone small high-gradient Gaussians, split
  large ones, prune low-amplitude or out-of-bounds primitives.  Optimizer is rebuilt
  with LR warmup after each densify step.
* **Mixed-precision training** (AMP) with proper float32 fallbacks for Cholesky /
  eigvalsh which do not support fp16.

Usage
-----
    python train_gmf.py                     # uses config.yml in cwd
    python train_gmf.py --config my.yml     # custom config path

The script expects a single-channel 3D TIFF volume (Z, Y, X) and produces
checkpoint .pt files containing the model state_dict.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import logging
from dataclasses import dataclass, field as dc_field
from datetime import datetime
from typing import Optional

import numpy as np
import tifffile as tiff
import yaml
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional CUDA extension
# ---------------------------------------------------------------------------
try:
    import gaussian_eval_cuda
    HAS_CUDA_EXTENSION = True
    print("✓ Custom CUDA extension loaded")
except ImportError:
    HAS_CUDA_EXTENSION = False
    print("✗ Custom CUDA extension not found — using PyTorch fallback")


# ---------------------------------------------------------------------------
# Custom autograd function wrapping the CUDA kernels
# ---------------------------------------------------------------------------
def _build_L_chol(log_scales: torch.Tensor, quaternions: torch.Tensor, eps: float = 1e-5):
    """
    Compute Cholesky factor from learnable (log_scales, quaternions).
    This function is differentiable through PyTorch autograd.

    Returns L such that  L Lᵀ = R diag(s²) Rᵀ + εI.
    """
    K = log_scales.shape[0]
    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    S2 = torch.diag_embed(scales ** 2)
    Sigma = R @ S2 @ R.transpose(-2, -1)
    Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
    return torch.linalg.cholesky(Sigma_reg.float())


class _GaussianEvalCUDA(torch.autograd.Function):
    """
    Wraps the CUDA gaussian_eval_cuda.forward / .backward kernels.

    The kernels operate on (x, means, L_chol, amplitudes) and produce (N,K).
    The backward kernel returns grad_x, grad_means, grad_amplitudes —
    but NOT grad_L_chol.

    To propagate gradients to log_scales and quaternions we:
      1) Compute grad_L_chol analytically from the Mahalanobis distance
      2) Recompute L_chol from (log_scales, quaternions) inside backward
         with autograd enabled, then call torch.autograd.grad to chain
         grad_L_chol → grad_log_scales, grad_quaternions.
    """

    @staticmethod
    def forward(ctx, x, means, log_scales, quaternions, log_amplitudes, L_chol_detached):
        """
        Args:
            x:                (N, 3) query points
            means:            (K, 3) Gaussian centres
            log_scales:       (K, 3) learnable log-scales
            quaternions:      (K, 4) learnable quaternions
            log_amplitudes:   (K,) learnable log-amplitudes
            L_chol_detached:  (K, 3, 3) precomputed Cholesky factor (detached)
        """
        amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        # CUDA forward: returns (N, K)
        vals_nk = gaussian_eval_cuda.forward(
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol_detached.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )

        # Sum over K → (N,)
        output = vals_nk.sum(dim=1)

        ctx.save_for_backward(
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol_detached, amplitudes, vals_nk,
        )
        return output.to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output_n):
        (
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol, amplitudes, vals_nk,
        ) = ctx.saved_tensors
        N, K = vals_nk.shape

        # Expand (N,) → (N, K) since output = vals_nk.sum(dim=1)
        grad_nk = grad_output_n[:, None].expand(N, K).contiguous()

        # CUDA backward: returns [grad_x, grad_means, grad_L_chol, grad_amplitudes]
        # The kernel already computes grad_L_chol via analytic differentiation
        # of the forward substitution, accumulated over all N points per Gaussian.
        cuda_grads = gaussian_eval_cuda.backward(
            grad_nk.float(),
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol.contiguous().float(),
            amplitudes.detach().contiguous().float(),
            vals_nk.contiguous().float(),
        )
        grad_x = cuda_grads[0]               # (N, 3)
        grad_means = cuda_grads[1]            # (K, 3)
        grad_L_chol = cuda_grads[2]           # (K, 3, 3) — from CUDA kernel
        grad_amplitudes_raw = cuda_grads[3]   # (K,)

        # Ensure only lower-triangular entries are used
        grad_L_chol = torch.tril(grad_L_chol)

        # --- Chain grad_L_chol → grad_log_scales, grad_quaternions ---
        # Recompute L_chol from (log_scales, quaternions) WITH autograd,
        # then use torch.autograd.grad to propagate.
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # grad_log_amplitudes: chain rule through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amplitudes

        # Return grads for: x, means, log_scales, quaternions, log_amplitudes, L_chol
        return (
            grad_x.to(x.dtype),
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
            None,  # L_chol_detached — no grad needed
        )


_gaussian_eval_cuda_fn = _GaussianEvalCUDA.apply


# Gradient supervision uses direct field evaluation with PyTorch autograd
# The field() call will automatically use CUDA kernels when available

class _GradientSupervisionCUDA(torch.autograd.Function):
    """
    Fused CUDA gradient supervision with custom backward.
    
    Forward: uses CUDA kernel to evaluate field at 4 points and compute L1 loss.
    Backward: uses CUDA kernel to compute gradients w.r.t. means, L_chol, amplitudes,
    then chains L_chol gradients to log_scales and quaternions via PyTorch autograd.
    """

    @staticmethod
    def forward(ctx, x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
                means, log_scales, quaternions, log_amplitudes):
        L_chol = _build_L_chol(log_scales, quaternions)
        amps = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        # CUDA forward: returns [grad_loss (N,), pred_sums (N, 4)]
        results = gaussian_eval_cuda.gradient_supervision(
            x_center.contiguous().float(),
            x_dx.contiguous().float(),
            x_dy.contiguous().float(),
            x_dz.contiguous().float(),
            v_center.contiguous().float(),
            v_dx.contiguous().float(),
            v_dy.contiguous().float(),
            v_dz.contiguous().float(),
            means.contiguous().float(),
            L_chol.detach().contiguous().float(),
            amps.detach().contiguous().float(),
        )
        grad_loss = results[0]   # (N,)
        pred_sums = results[1]   # (N, 4)

        ctx.save_for_backward(
            x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
            means, log_scales, quaternions, log_amplitudes,
            L_chol, amps, pred_sums,
        )
        return grad_loss

    @staticmethod
    def backward(ctx, grad_output):
        (x_center, x_dx, x_dy, x_dz, v_center, v_dx, v_dy, v_dz,
         means, log_scales, quaternions, log_amplitudes,
         L_chol, amps, pred_sums) = ctx.saved_tensors

        # CUDA backward: returns [grad_means, grad_L (K,3,3), grad_amplitudes]
        cuda_grads = gaussian_eval_cuda.gradient_supervision_backward(
            grad_output.contiguous().float(),
            x_center.contiguous().float(),
            x_dx.contiguous().float(),
            x_dy.contiguous().float(),
            x_dz.contiguous().float(),
            v_center.contiguous().float(),
            v_dx.contiguous().float(),
            v_dy.contiguous().float(),
            v_dz.contiguous().float(),
            means.contiguous().float(),
            L_chol.detach().contiguous().float(),
            amps.detach().contiguous().float(),
            pred_sums.contiguous().float(),
        )
        grad_means = cuda_grads[0]           # (K, 3)
        grad_L_chol = torch.tril(cuda_grads[1])  # (K, 3, 3)
        grad_amplitudes_raw = cuda_grads[2]  # (K,)

        # Chain grad_L_chol → grad_log_scales, grad_quaternions
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # Chain grad_amplitudes through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amps

        return (
            None, None, None, None,  # x_center, x_dx, x_dy, x_dz
            None, None, None, None,  # v_center, v_dx, v_dy, v_dz
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
        )


_gradient_supervision_cuda_fn = _GradientSupervisionCUDA.apply


class _AnalyticalFieldGradCUDA(torch.autograd.Function):
    """
    Computes the analytical spatial gradient ∇_x f(x) of the Gaussian field.

    ∇_x f(x) = Σ_k -v_k · L_k^{-T} y_k, where y_k = L_k^{-1}(x - μ_k)

    This is computed alongside f(x) in a single CUDA kernel, but only the
    gradient output (N, 3) participates in the loss. The reconstruction loss
    goes through the existing _GaussianEvalCUDA path separately.
    """

    @staticmethod
    def forward(ctx, x, means, log_scales, quaternions, log_amplitudes, L_chol_detached):
        amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

        results = gaussian_eval_cuda.forward_with_field_grad(
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol_detached.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )
        # results[0] = val (N,), results[1] = field_grad (N, 3)
        field_grad = results[1]

        ctx.save_for_backward(
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol_detached, amplitudes,
        )
        return field_grad

    @staticmethod
    def backward(ctx, grad_field_grad):
        (
            x, means, log_scales, quaternions, log_amplitudes,
            L_chol, amplitudes,
        ) = ctx.saved_tensors

        cuda_grads = gaussian_eval_cuda.analytical_grad_backward(
            grad_field_grad.contiguous().float(),
            x.contiguous().float(),
            means.contiguous().float(),
            L_chol.contiguous().float(),
            amplitudes.detach().contiguous().float(),
        )
        grad_means = cuda_grads[0]                   # (K, 3)
        grad_L_chol = torch.tril(cuda_grads[1])      # (K, 3, 3)
        grad_amplitudes_raw = cuda_grads[2]           # (K,)

        # Chain grad_L_chol → grad_log_scales, grad_quaternions
        with torch.enable_grad():
            ls = log_scales.detach().requires_grad_(True)
            qt = quaternions.detach().requires_grad_(True)
            L_recomp = _build_L_chol(ls, qt)
            grads = torch.autograd.grad(
                L_recomp, [ls, qt],
                grad_outputs=grad_L_chol,
                allow_unused=True,
            )
            grad_log_scales = grads[0]
            grad_quaternions = grads[1]

        # Chain grad_amplitudes through exp(clamp(log_amp))
        grad_log_amplitudes = grad_amplitudes_raw * amplitudes

        return (
            None,  # x
            grad_means.to(means.dtype),
            grad_log_scales,
            grad_quaternions,
            grad_log_amplitudes.to(log_amplitudes.dtype),
            None,  # L_chol_detached
        )


_analytical_field_grad_cuda_fn = _AnalyticalFieldGradCUDA.apply


# ===================================================================
#  Utilities
# ===================================================================
def load_config(path: str = "config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{ts}.log")

    # Reset root logger to avoid duplicate handlers across runs
    root = logging.getLogger()
    root.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger("gmf_train")
    logger.info(f"Log → {log_file}")
    return logger


def load_tif_data(file_path: str) -> np.ndarray:
    """Load a 3-D TIFF and normalise to [0, 1] float32."""
    vol = tiff.imread(file_path).astype(np.float32)  # (Z, Y, X)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def load_swc(file_path: str) -> np.ndarray:
    """
    Load an SWC morphology file and return (N, 4) array: [x, y, z, radius].
    SWC format: id  type  x  y  z  radius  parent_id
    Skips comment lines starting with '#'.
    """
    rows = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 7:
                x, y, z, r = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                rows.append([x, y, z, r])
    if not rows:
        raise ValueError(f"No valid SWC nodes found in {file_path}")
    return np.array(rows, dtype=np.float32)


def swc_to_normalised_coords(
    swc_data: np.ndarray,
    vol_shape: tuple[int, int, int],
    bounds: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert SWC coordinates (in voxel space) to normalised [-1, 1] coords.

    Args:
        swc_data: (N, 4) array [x, y, z, radius] in voxel coordinates.
        vol_shape: (Z, Y, X) shape of the volume.
        bounds: [[xlo,xhi],[ylo,yhi],[zlo,zhi]] normalised bounds (default [-1,1]).

    Returns:
        coords: (N, 3) normalised [x, y, z] coords.
        radii:  (N,) normalised radii (average of xyz scale factors).
    """
    if bounds is None:
        bounds = [[-1, 1], [-1, 1], [-1, 1]]
    Z, Y, X = vol_shape
    # SWC x → volume X axis, y → Y, z → Z
    vox_max = np.array([X - 1, Y - 1, Z - 1], dtype=np.float32)
    vox_max = np.maximum(vox_max, 1.0)  # avoid /0

    xyz = swc_data[:, :3]  # (N, 3) in voxel coords
    # Normalise each axis to [0, 1] then map to bounds
    norm01 = xyz / vox_max  # (N, 3) in [0, 1]
    coords = np.zeros_like(norm01)
    scale_factors = []
    for i in range(3):
        lo, hi = bounds[i][0], bounds[i][1]
        coords[:, i] = norm01[:, i] * (hi - lo) + lo
        scale_factors.append((hi - lo) / vox_max[i])

    # Normalise radius: average scale factor across axes
    avg_scale = np.mean(scale_factors)
    radii = swc_data[:, 3] * avg_scale

    return coords.astype(np.float32), radii.astype(np.float32)


# ===================================================================
#  Loss helpers
# ===================================================================
def charbonnier(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Charbonnier penalty  sqrt(x² + ε²).  eps=1e-6 keeps floor negligible."""
    return torch.sqrt(x * x + eps * eps)


# ===================================================================
#  GPU volume sampling
# ===================================================================
_sampling_cdf_cache: dict[str, torch.Tensor] = {}


def _flat_indices_gpu(
    vol_gpu: torch.Tensor,
    num_samples: int,
    intensity_weighted: bool,
    cache_key: str | None,
) -> torch.Tensor:
    Z, Y, X = vol_gpu.shape
    Nvox = Z * Y * X
    device = vol_gpu.device

    if num_samples > Nvox:
        raise ValueError(f"num_samples={num_samples} > total voxels={Nvox}")

    if intensity_weighted:
        if cache_key and cache_key in _sampling_cdf_cache:
            cdf = _sampling_cdf_cache[cache_key]
        else:
            flat = vol_gpu.reshape(-1)
            probs = flat / (flat.sum() + 1e-12)
            cdf = torch.cumsum(probs, dim=0)
            if cache_key:
                _sampling_cdf_cache[cache_key] = cdf
        u = torch.rand(num_samples, device=device)
        idx = torch.searchsorted(cdf, u).clamp(0, Nvox - 1)
    else:
        idx = torch.randperm(Nvox, device=device)[:num_samples]
    return idx


def _idx_to_coords(idx: torch.Tensor, Z: int, Y: int, X: int):
    """Flat index → integer (z, y, x) + normalised [-1, 1] coords."""
    z = idx // (Y * X)
    rem = idx % (Y * X)
    y = rem // X
    x = rem % X

    xn = (x.float() / max(X - 1, 1)) * 2 - 1
    yn = (y.float() / max(Y - 1, 1)) * 2 - 1
    zn = (z.float() / max(Z - 1, 1)) * 2 - 1
    return z, y, x, xn, yn, zn


def sample_points_from_volume(
    vol_gpu: torch.Tensor,
    num_samples: int,
    intensity_weighted: bool = True,
    cache_key: str | None = None,
):
    """Sample random voxels and return (pts (N,3), vals (N,))."""
    Z, Y, X = vol_gpu.shape
    idx = _flat_indices_gpu(vol_gpu, num_samples, intensity_weighted, cache_key)
    z, y, x, xn, yn, zn = _idx_to_coords(idx, Z, Y, X)
    pts = torch.stack([xn, yn, zn], dim=1)
    vals = vol_gpu[z, y, x]
    return pts, vals


def sample_points_with_neighbors(
    vol_gpu: torch.Tensor,
    num_samples: int,
    delta_vox: int = 1,
    intensity_weighted: bool = True,
    cache_key: str | None = None,
):
    """
    Sample centre voxels *and* their +δ neighbours along each axis for
    finite-difference gradient supervision.

    Returns
    -------
    pts, vals,              — centre points
    pts_dx, vals_dx,        — neighbour shifted in x
    pts_dy, vals_dy,        — neighbour shifted in y
    pts_dz, vals_dz         — neighbour shifted in z
    """
    Z, Y, X = vol_gpu.shape
    device = vol_gpu.device

    idx = _flat_indices_gpu(vol_gpu, num_samples, intensity_weighted, cache_key)
    z, y, x, xn, yn, zn = _idx_to_coords(idx, Z, Y, X)

    pts = torch.stack([xn, yn, zn], dim=1)
    vals = vol_gpu[z, y, x]

    # Forward neighbours (clamped at boundary)
    x1 = (x + delta_vox).clamp(0, X - 1)
    y1 = (y + delta_vox).clamp(0, Y - 1)
    z1 = (z + delta_vox).clamp(0, Z - 1)

    vals_dx = vol_gpu[z, y, x1]
    vals_dy = vol_gpu[z, y1, x]
    vals_dz = vol_gpu[z1, y, x]

    # Neighbour normalised coords
    x1n = (x1.float() / max(X - 1, 1)) * 2 - 1
    y1n = (y1.float() / max(Y - 1, 1)) * 2 - 1
    z1n = (z1.float() / max(Z - 1, 1)) * 2 - 1

    pts_dx = torch.stack([x1n, yn, zn], dim=1)
    pts_dy = torch.stack([xn, y1n, zn], dim=1)
    pts_dz = torch.stack([xn, yn, z1n], dim=1)

    return pts, vals, pts_dx, vals_dx, pts_dy, vals_dy, pts_dz, vals_dz


# ===================================================================
#  MIP helpers
# ===================================================================
def mip_teacher_z(vol: np.ndarray) -> np.ndarray:
    """Ground-truth z-axis Maximum Intensity Projection."""
    return vol.max(axis=0).astype(np.float32)


def sample_pixels_from_mip(mip: np.ndarray, num_samples: int):
    Y, X = mip.shape
    Npix = Y * X
    if num_samples > Npix:
        raise ValueError(f"num_samples={num_samples} > total pixels={Npix}")
    idx = np.random.choice(Npix, size=num_samples, replace=False)
    y, x = idx // X, idx % X
    xn = (x / max(X - 1, 1)) * 2 - 1
    yn = (y / max(Y - 1, 1)) * 2 - 1
    xy = torch.from_numpy(np.stack([xn, yn], axis=1)).float()
    t = torch.from_numpy(mip[y, x]).float()
    return xy, t


def compute_tau_schedule(tau_start: float, tau_end: float, t: float) -> float:
    """Anneal soft-max temperature.  t ∈ [0, 1]."""
    return float(tau_start * (tau_end / max(tau_start, 1e-12)) ** t)


# ===================================================================
#  Regularisers
# ===================================================================
def tubular_regulariser(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Encourage tubular shapes: penalise (λ1+λ2)/λ3 being large."""
    eig = torch.linalg.eigvalsh(Sigma.float())        # (K, 3) ascending
    eig = torch.sort(eig, dim=-1)[0]
    return ((eig[:, 0] + eig[:, 1]) / (eig[:, 2] + eps)).mean()


def cross_section_symmetry_reg(Sigma: torch.Tensor) -> torch.Tensor:
    """Encourage circular cross-sections: penalise |λ1 − λ2|."""
    eig = torch.linalg.eigvalsh(Sigma.float())
    eig = torch.sort(eig, dim=-1)[0]
    return (eig[:, 0] - eig[:, 1]).abs().mean()


def _gradient_magnitudes(param: nn.Parameter) -> torch.Tensor:
    if param.grad is None:
        return torch.zeros(param.shape[0], device=param.device)
    if param.grad.ndim > 1:
        return torch.norm(param.grad, dim=-1)
    return param.grad.abs()


# ===================================================================
#  Model
# ===================================================================
class GaussianMixtureField(nn.Module):
    """
    Anisotropic 3-D Gaussian Mixture Field.

    Parameters are (means, log_scales, quaternions, log_amplitudes) and the
    covariance is  Σ = R diag(s²) Rᵀ  with R from unit quaternion.
    """

    def __init__(
        self,
        num_gaussians: int,
        init_scale: float = 0.05,
        init_amplitude: float = 0.1,
        bounds: list | None = None,
        aabb: list | None = None,
        swc_coords: np.ndarray | None = None,
        swc_radii: np.ndarray | None = None,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians

        # Axis-aligned bounding box
        if aabb is not None:
            self.aabb = torch.tensor(aabb, dtype=torch.float32)
        elif bounds is not None:
            self.aabb = torch.tensor(bounds, dtype=torch.float32)
        else:
            self.aabb = torch.tensor([[-1, 1], [-1, 1], [-1, 1]], dtype=torch.float32)

        # --- initialise means ---
        if swc_coords is not None:
            # Initialise from SWC neuron morphology
            n_swc = swc_coords.shape[0]
            if n_swc >= num_gaussians:
                # Subsample: uniformly pick num_gaussians points along the skeleton
                idx = np.linspace(0, n_swc - 1, num_gaussians, dtype=int)
                means = torch.from_numpy(swc_coords[idx]).float()
            else:
                # Fewer SWC nodes than Gaussians: use all nodes + fill rest
                # by interpolating random pairs along skeleton edges
                means_swc = torch.from_numpy(swc_coords).float()
                n_extra = num_gaussians - n_swc
                # Random pairs of consecutive nodes for interpolation
                pair_idx = torch.randint(0, max(n_swc - 1, 1), (n_extra,))
                t = torch.rand(n_extra, 1)
                extra = means_swc[pair_idx] * (1 - t) + means_swc[pair_idx + 1] * t
                # Add small jitter to avoid exact duplicates
                extra += torch.randn_like(extra) * 0.001
                means = torch.cat([means_swc, extra], dim=0)
            print(f"SWC init: {n_swc} nodes → {num_gaussians} Gaussians")
        elif bounds is not None:
            means = torch.zeros(num_gaussians, 3)
            for i in range(3):
                lo, hi = bounds[i][0], bounds[i][1]
                means[:, i] = torch.rand(num_gaussians) * (hi - lo) + lo
        else:
            means = torch.randn(num_gaussians, 3) * 0.1

        self.means = nn.Parameter(means)

        # Scale init: use SWC radii if provided, otherwise use init_scale
        if swc_radii is not None:
            # Per-Gaussian scale from SWC radius (isotropic initial scale)
            if swc_coords.shape[0] >= num_gaussians:
                idx = np.linspace(0, swc_coords.shape[0] - 1, num_gaussians, dtype=int)
                radii_sel = swc_radii[idx]
            else:
                # Pad extra Gaussians with median radius
                med_r = float(np.median(swc_radii))
                radii_sel = np.concatenate([
                    swc_radii,
                    np.full(num_gaussians - swc_coords.shape[0], med_r, dtype=np.float32)
                ])
            radii_t = torch.from_numpy(radii_sel).float().clamp(min=1e-4)
            self.log_scales = nn.Parameter(
                torch.log(radii_t).unsqueeze(-1).expand(-1, 3).contiguous()
            )
            print(f"SWC scale init: radius range [{radii_t.min():.4f}, {radii_t.max():.4f}]")
        else:
            # Fallback: auto-compute if init_scale too small
            if init_scale < 1e-3:
                side = 2.0  # [-1, 1]
                init_scale = side / (num_gaussians ** (1.0 / 3.0)) * 1.5
                print(f"⚠ init_scale too small, auto-set to {init_scale:.4f}")
            self.log_scales = nn.Parameter(
                torch.ones(num_gaussians, 3) * math.log(init_scale)
            )

        q = torch.zeros(num_gaussians, 4)
        q[:, 0] = 1.0  # identity rotation
        self.quaternions = nn.Parameter(q)

        # Amplitude init: start moderate, not at 1.0
        # With many overlapping Gaussians, amp=1.0 creates huge peaks;
        # amp=0.01–0.1 lets them sum to reasonable values.
        self.log_amplitudes = nn.Parameter(
            torch.ones(num_gaussians) * math.log(max(init_amplitude, 1e-6))
        )

    # ---- geometry helpers ------------------------------------------------
    @staticmethod
    def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return R

    def get_covariance_matrices(self) -> torch.Tensor:
        s = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        R = self.quat_to_rotmat(self.quaternions)
        S2 = torch.diag_embed(s ** 2)
        return R @ S2 @ R.transpose(-2, -1)

    # ---- constraints (call after optimizer.step) -------------------------
    def apply_aabb_clamp(self, margin: float = 0.0):
        aabb = self.aabb.to(self.means.device)
        with torch.no_grad():
            for i in range(3):
                self.means.data[:, i].clamp_(aabb[i, 0] + margin, aabb[i, 1] - margin)

    def clamp_log_scales_(self, lo: float, hi: float):
        with torch.no_grad():
            self.log_scales.data.clamp_(lo, hi)

    def clamp_log_amplitudes_(self, lo: float, hi: float):
        with torch.no_grad():
            self.log_amplitudes.data.clamp_(lo, hi)

    # ---- forward (dispatches CUDA kernel or K-chunked PyTorch) -----------
    def forward(self, x: torch.Tensor, k_chunk: int = 1024) -> torch.Tensor:
        """
        Evaluate field at query points  x  (N, 3) → (N,).

        If the custom CUDA extension is available and the full (N, K) tensor
        fits in GPU memory, uses the fused CUDA kernel for speedup on the
        forward + backward of (x, means, amplitudes), while gradients for
        log_scales and quaternions are chained through PyTorch autograd
        via the Cholesky factorisation.

        Otherwise falls back to K-chunked PyTorch solve_triangular.
        """
        N = x.shape[0]
        K = self.num_gaussians

        # --- CUDA kernel path ---
        if HAS_CUDA_EXTENSION and x.is_cuda:
            # Memory estimate: (N*K) * ~48 bytes (vals + recompute in backward)
            mem_estimate = N * K * 48
            mem_free = torch.cuda.mem_get_info(x.device)[0]
            use_cuda = mem_estimate < mem_free * 0.5

            if use_cuda:
                if not hasattr(self, '_cuda_logged'):
                    print(f"✓ Using CUDA kernel: N={N}, K={K}, mem_estimate={mem_estimate/1e9:.2f}GB")
                    self._cuda_logged = True
                # Precompute L_chol (detached) — the autograd Function
                # recomputes it inside backward with grad tracking.
                with torch.no_grad():
                    L_chol = _build_L_chol(self.log_scales, self.quaternions)
                return _gaussian_eval_cuda_fn(
                    x, self.means, self.log_scales, self.quaternions,
                    self.log_amplitudes, L_chol,
                )
            else:
                if not hasattr(self, '_fallback_logged'):
                    print(f"✗ CUDA kernel skipped (memory): N={N}, K={K}, need={mem_estimate/1e9:.2f}GB, free={mem_free/1e9:.2f}GB")
                    self._fallback_logged = True

        # --- Chunked PyTorch fallback ---
        dtype_in = x.dtype
        amps = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0))

        Sigma = self.get_covariance_matrices()
        eps = 1e-5
        Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device).unsqueeze(0)
        try:
            L = torch.linalg.cholesky(Sigma_reg.float())
        except torch._C._LinAlgError:
            Sigma_reg = Sigma + 1e-3 * torch.eye(3, device=Sigma.device).unsqueeze(0)
            L = torch.linalg.cholesky(Sigma_reg.float())

        out = torch.zeros(N, device=x.device, dtype=dtype_in)

        for ks in range(0, K, k_chunk):
            ke = min(ks + k_chunk, K)
            G = ke - ks

            mu = self.means[ks:ke]
            a = amps[ks:ke]
            Lc = L[ks:ke]

            diff = (x[:, None, :] - mu[None, :, :])
            diff_flat = diff.reshape(N * G, 3, 1).float()
            L_exp = Lc.unsqueeze(0).expand(N, G, 3, 3).reshape(N * G, 3, 3)

            y = torch.linalg.solve_triangular(L_exp, diff_flat, upper=False)
            mahal = (y.squeeze(-1) ** 2).sum(-1).reshape(N, G)

            vals = a[None, :] * torch.exp(-0.5 * mahal.to(dtype_in))
            out = out + vals.sum(dim=1)

        return out

    # ---- densify / prune -------------------------------------------------
    def densify_and_prune(
        self,
        grad_threshold: float = 1.5e-4,
        min_opacity: float = 5e-4,
        max_scale: float = 0.8,
        split_scale_threshold: float = 0.05,
        enforce_aabb: bool = True,
        max_gaussians: int = 0,
        max_clones: int = 0,
    ) -> dict:
        with torch.no_grad():
            device = self.means.device

            grad_mag = _gradient_magnitudes(self.means)
            scales = torch.exp(self.log_scales).clamp(1e-5, 1e2)
            max_s = scales.max(dim=-1)[0]
            amps = torch.exp(self.log_amplitudes)

            high_grad = grad_mag > grad_threshold
            small = max_s < split_scale_threshold
            clone_mask = high_grad & small
            split_mask = high_grad & (~small)

            new_m, new_ls, new_q, new_la = [], [], [], []

            # Clone (with optional cap)
            if clone_mask.any():
                if max_clones > 0 and int(clone_mask.sum()) > max_clones:
                    # Keep only the top-gradient clones
                    clone_idx = clone_mask.nonzero(as_tuple=True)[0]
                    clone_grads = grad_mag[clone_idx]
                    _, topk = clone_grads.topk(max_clones, largest=True)
                    clone_idx = clone_idx[topk]
                    clone_mask = torch.zeros_like(clone_mask)
                    clone_mask[clone_idx] = True
                new_m.append(self.means[clone_mask])
                new_ls.append(self.log_scales[clone_mask])
                new_q.append(self.quaternions[clone_mask])
                new_la.append(self.log_amplitudes[clone_mask])

            # Split
            if split_mask.any():
                m = self.means[split_mask]
                ls = self.log_scales[split_mask]
                q = self.quaternions[split_mask]
                la = self.log_amplitudes[split_mask]
                R = self.quat_to_rotmat(q)
                s = torch.exp(ls)
                imax = s.argmax(dim=-1)
                bi = torch.arange(m.shape[0], device=device)
                principal = R[bi, :, imax]
                offset = s[bi, imax].unsqueeze(-1) * 0.5

                c1 = m + principal * offset
                c2 = m - principal * offset
                child_ls = ls - math.log(1.6)
                child_la = la - math.log(2.0)

                new_m.extend([c1, c2])
                new_ls.extend([child_ls, child_ls])
                new_q.extend([q, q])
                new_la.extend([child_la, child_la])

            # Prune
            keep = (amps > min_opacity) & (max_s < max_scale)
            if enforce_aabb:
                aabb = self.aabb.to(device)
                within = torch.ones(self.means.shape[0], dtype=torch.bool, device=device)
                for i in range(3):
                    within &= (self.means[:, i] >= aabb[i, 0]) & (
                        self.means[:, i] <= aabb[i, 1]
                    )
                keep &= within

            old_K = self.num_gaussians
            pruned = int((~keep).sum().item())

            parts = [
                self.means[keep],
                self.log_scales[keep],
                self.quaternions[keep],
                self.log_amplitudes[keep],
            ]
            if new_m:
                parts_new = [
                    torch.cat(new_m),
                    torch.cat(new_ls),
                    torch.cat(new_q),
                    torch.cat(new_la),
                ]
                combined = [torch.cat([p, pn]) for p, pn in zip(parts, parts_new)]
            else:
                combined = parts

            # Enforce max_gaussians cap: keep highest-amplitude Gaussians
            cap_pruned = 0
            if max_gaussians > 0 and combined[0].shape[0] > max_gaussians:
                cap_pruned = combined[0].shape[0] - max_gaussians
                cap_amps = torch.exp(combined[3])
                _, topk_idx = cap_amps.topk(max_gaussians, largest=True)
                topk_idx = topk_idx.sort()[0]
                combined = [c[topk_idx] for c in combined]

            self.means = nn.Parameter(combined[0])
            self.log_scales = nn.Parameter(combined[1])
            self.quaternions = nn.Parameter(combined[2])
            self.log_amplitudes = nn.Parameter(combined[3])
            self.num_gaussians = int(combined[0].shape[0])

            return {
                "old": old_K,
                "new": self.num_gaussians,
                "pruned": pruned,
                "cloned": int(clone_mask.sum().item()),
                "split": int(split_mask.sum().item()) * 2,
                "Δ": self.num_gaussians - old_K,
                "cap_pruned": cap_pruned,
            }


# ===================================================================
#  Loss functions
# ===================================================================
def loss_volume(
    field: GaussianMixtureField,
    x: torch.Tensor,
    v: torch.Tensor,
    # neighbour data (optional — pass None to skip gradient loss)
    x_dx: torch.Tensor | None = None,
    v_dx: torch.Tensor | None = None,
    x_dy: torch.Tensor | None = None,
    v_dy: torch.Tensor | None = None,
    x_dz: torch.Tensor | None = None,
    v_dz: torch.Tensor | None = None,
    *,
    w_grad: float = 0.3,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    w_scale: float = 5e-4,
    scale_target: float | None = 0.03,
) -> tuple[torch.Tensor, dict]:

    pred = field(x)
    l_rec = F.mse_loss(pred, v)

    # --- gradient supervision (finite differences) ---
    l_grad = torch.zeros((), device=x.device)
    if x_dx is not None and w_grad > 0:
        use_cuda_grad = (
            HAS_CUDA_EXTENSION
            and x.is_cuda
            and hasattr(gaussian_eval_cuda, 'gradient_supervision_backward')
        )

        use_analytical_grad = (
            HAS_CUDA_EXTENSION
            and x.is_cuda
            and hasattr(gaussian_eval_cuda, 'analytical_grad_backward')
        )

        if use_analytical_grad:
            # Analytical gradient: compute ∇_x f(x) in single kernel call
            L_chol = _build_L_chol(field.log_scales, field.quaternions).detach()
            field_grad = _analytical_field_grad_cuda_fn(
                x, field.means, field.log_scales, field.quaternions,
                field.log_amplitudes, L_chol,
            )  # (N, 3)

            # Ground truth finite differences from volume
            # field_grad is the derivative w.r.t. normalised coords;
            # multiply by the step in normalised coords to predict the difference
            deltas = torch.stack([
                x_dx[:, 0] - x[:, 0],
                x_dy[:, 1] - x[:, 1],
                x_dz[:, 2] - x[:, 2],
            ], dim=-1)  # (N, 3)

            pred_diff = field_grad * deltas  # predicted finite differences (N, 3)
            gt_diff = torch.stack([v_dx - v, v_dy - v, v_dz - v], dim=-1)  # (N, 3)
            l_grad = F.l1_loss(pred_diff, gt_diff)
        elif use_cuda_grad:
            # Fused CUDA kernel with custom backward — fastest path
            grad_loss_per_point = _gradient_supervision_cuda_fn(
                x, x_dx, x_dy, x_dz,
                v, v_dx, v_dy, v_dz,
                field.means, field.log_scales, field.quaternions, field.log_amplitudes
            )
            l_grad = grad_loss_per_point.mean()
        else:
            # PyTorch fallback: match SIGNED gradients (preserves edge direction)
            p_dx = field(x_dx)
            p_dy = field(x_dy)
            p_dz = field(x_dz)
            l_grad = (
                F.l1_loss(p_dx - pred, v_dx - v)  # signed gradient in x
                + F.l1_loss(p_dy - pred, v_dy - v)  # signed gradient in y
                + F.l1_loss(p_dz - pred, v_dz - v)  # signed gradient in z
            )

    # --- covariance regularisers ---
    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    # --- scale regulariser (prevent blobs) ---
    scales = torch.exp(field.log_scales).clamp(1e-6, 1e2)
    if scale_target is not None:
        l_scale = F.relu(scales - scale_target).mean()
    else:
        l_scale = scales.mean()

    total = (
        l_rec
        + w_grad * l_grad
        + w_tube * l_tube
        + w_cross * l_csym
        + w_scale * l_scale
    )
    parts = {
        "rec": l_rec,
        "grad": l_grad,
        "tube": l_tube,
        "csym": l_csym,
        "scale": l_scale,
    }
    return total, parts


def render_soft_mip_z(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    n_z: int,
    tau: float,
    pt_chunk: int = 16384,
) -> torch.Tensor:
    """Soft z-MIP via LogSumExp along z-rays."""
    device = xy.device
    P = xy.shape[0]
    z_vals = torch.linspace(-1, 1, n_z, device=device, dtype=xy.dtype)

    pts = torch.cat(
        [
            xy[:, None, :].expand(P, n_z, 2),
            z_vals[None, :, None].expand(P, n_z, 1),
        ],
        dim=-1,
    ).reshape(-1, 3)  # (P*n_z, 3)

    vals = []
    for i in range(0, pts.shape[0], pt_chunk):
        vals.append(field(pts[i : i + pt_chunk]))
    v = torch.cat(vals).reshape(P, n_z)

    tau_safe = max(tau, 1e-6)
    return tau_safe * torch.logsumexp(v / tau_safe, dim=1)


def loss_mip(
    field: GaussianMixtureField,
    xy: torch.Tensor,
    mip_gt: torch.Tensor,
    n_z: int,
    tau: float,
    *,
    w_tube: float = 1e-4,
    w_cross: float = 1e-4,
    mip_batch: int = 512,
) -> tuple[torch.Tensor, dict]:
    P = xy.shape[0]
    if P <= mip_batch:
        pred = render_soft_mip_z(field, xy, n_z, tau)
    else:
        chunks = []
        for i in range(0, P, mip_batch):
            chunks.append(render_soft_mip_z(field, xy[i : i + mip_batch], n_z, tau))
        pred = torch.cat(chunks)

    l_img = F.l1_loss(pred, mip_gt)

    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regulariser(Sigma)
    l_csym = cross_section_symmetry_reg(Sigma)

    total = l_img + w_tube * l_tube + w_cross * l_csym
    return total, {"mip": l_img, "tube": l_tube, "csym": l_csym}


# ===================================================================
#  Weight schedule
# ===================================================================
def weight_schedule(cfg: dict, step: int, total: int) -> tuple[float, float]:
    sch = cfg["training"].get("weight_schedule", "constant").lower()
    if sch == "constant":
        return (
            float(cfg["training"].get("w_vol", 1.0)),
            float(cfg["training"].get("w_mip", 1.0)),
        )

    vs = float(cfg["training"].get("w_vol_start", 1.0))
    ms = float(cfg["training"].get("w_mip_start", 0.1))
    ve = float(cfg["training"].get("w_vol_end", 1.0))
    me = float(cfg["training"].get("w_mip_end", 1.0))
    tf = float(cfg["training"].get("weight_transition_fraction", 0.3))

    t = step / max(1, total - 1)

    if sch == "step":
        return (vs, ms) if t < tf else (ve, me)

    if sch == "linear_ramp":
        if t < tf:
            return vs, ms
        r = (t - tf) / max(1e-12, 1.0 - tf)
        return vs + (ve - vs) * r, ms + (me - ms) * r

    return float(cfg["training"].get("w_vol", 1.0)), float(
        cfg["training"].get("w_mip", 1.0)
    )


# ===================================================================
#  Training loop
# ===================================================================
def train(
    field: GaussianMixtureField,
    vol: np.ndarray,
    cfg: dict,
    device: str,
    log_dir: str,
) -> GaussianMixtureField:
    logger = setup_logger(log_dir)
    field = field.to(device)
    field.train()

    tc = cfg["training"]  # shorthand
    mode = tc.get("mode", "volume").lower()
    steps = int(tc.get("steps", 20000))
    lr = float(tc.get("learning_rate", 1e-2))

    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    lr_min_frac = float(tc.get("lr_min_fraction", 0.01))  # decay to 1% of initial LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=lr * lr_min_frac
    )

    use_amp = bool(tc.get("mixed_precision", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # -- sampling ---------------------------------------------------------
    vol_pts = int(tc.get("vol_points_per_step", 8192))
    iwt = bool(tc.get("vol_intensity_weighted", True))

    use_grad = bool(tc.get("use_grad_loss", True))
    delta_vox = int(tc.get("grad_delta_vox", 1))

    # -- loss weights -----------------------------------------------------
    w_grad = float(tc.get("lambda_grad", 0.3))
    w_tube = float(tc.get("lambda_tube", 1e-4))
    w_cross = float(tc.get("lambda_cross", 1e-4))
    w_scale = float(tc.get("lambda_scale", 5e-4))
    scale_target = tc.get("scale_target", 0.03)
    if scale_target is not None:
        scale_target = float(scale_target)

    # -- MIP --------------------------------------------------------------
    mip_px = int(tc.get("mip_pixels_per_step", 4096))
    n_z = int(tc.get("mip_z_samples", 128))
    tau_s = float(tc.get("tau_start", 0.08))
    tau_e = float(tc.get("tau_end", 0.02))
    mip_img = mip_teacher_z(vol)

    # -- amplitude clamping ------------------------------------------------
    clamp_amp = bool(tc.get("clamp_amplitudes", True))
    la_min = float(tc.get("log_amp_min", math.log(1e-4)))   # exp(-9.2) ≈ 0.0001
    la_max = float(tc.get("log_amp_max", math.log(1.0)))    # exp(0) = 1.0
    if clamp_amp:
        logger.info(f"Amplitude clamp=[{la_min:.3f}, {la_max:.3f}]  (amp=[{math.exp(la_min):.5f}, {math.exp(la_max):.4f}])")

    # -- gradient clipping -------------------------------------------------
    grad_clip = float(tc.get("grad_clip_norm", 1.0))  # 0 = disabled
    if grad_clip > 0:
        logger.info(f"Gradient clipping: max_norm={grad_clip}")

    # -- scale clamping ---------------------------------------------------
    do_clamp = bool(tc.get("clamp_scales", True))
    ls_min = float(tc.get("log_scale_min", math.log(5e-4)))   # ~0.0005
    ls_max = float(tc.get("log_scale_max", math.log(0.3)))    # ~0.3 in normalised coords

    # Sanity check: warn if init_scale would be clamped immediately
    init_ls = math.log(float(cfg["model"].get("init_scale", 0.05)))
    if do_clamp and (init_ls < ls_min or init_ls > ls_max):
        logger.warning(
            f"init_scale log={init_ls:.3f} outside clamp range [{ls_min:.3f}, {ls_max:.3f}] "
            f"— Gaussians will be clamped at init! Consider widening clamp range."
        )

    # -- AABB hard clamp --------------------------------------------------
    aabb_hard = bool(tc.get("enforce_aabb_hard", False))

    # -- densify ----------------------------------------------------------
    dens_on = bool(tc.get("densify_enabled", False))
    dens_from = int(tc.get("densify_from_iter", 500))
    dens_until = int(tc.get("densify_until_iter", 25000))
    dens_every = int(tc.get("densify_interval", 100))
    dens_grad = float(tc.get("densify_grad_threshold", 1.5e-4))
    dens_minop = float(tc.get("densify_min_opacity", 5e-4))
    dens_maxsc = float(tc.get("densify_max_scale", 0.8))
    dens_split = float(tc.get("densify_split_scale_threshold", 0.05))
    dens_aabb = bool(tc.get("densify_enforce_aabb", True))
    dens_lr_fac = float(tc.get("densify_lr_factor", 0.2))
    dens_lr_warm = int(tc.get("densify_lr_warmup_steps", 25))
    dens_maxK = int(tc.get("max_gaussians", 20000))
    dens_max_clones = int(tc.get("densify_max_clones_per_step", 0))  # 0 = unlimited
    dens_cooldown = int(tc.get("densify_cooldown_evals", 5))  # skip ES checks after densify
    last_dens = -(10**9)

    # EMA accumulator for mean gradient magnitudes (more stable than single-step)
    grad_accum = torch.zeros(field.num_gaussians, device=device)
    grad_count = 0

    # -- progressive mode -------------------------------------------------
    prog_split = float(tc.get("progressive_split_frac", 0.3))

    # -- prepare GPU volume -----------------------------------------------
    if device != "cuda":
        raise RuntimeError(
            "This script expects CUDA.  Add a CPU fallback path if needed."
        )
    vol_gpu = torch.from_numpy(vol).float().to(device)

    # -- logging ----------------------------------------------------------
    logger.info(f"Device={device}  Mode={mode}  Steps={steps}  LR={lr}  AMP={use_amp}")
    logger.info(f"Volume {vol.shape}  MIP {mip_img.shape}")
    logger.info(
        f"Losses: grad={use_grad}(w={w_grad}), tube={w_tube}, cross={w_cross}, "
        f"scale={w_scale}(target={scale_target})"
    )
    logger.info(f"Scale clamp={do_clamp}  [{ls_min:.3f}, {ls_max:.3f}]")
    logger.info(f"Densify={dens_on}  from={dens_from} until={dens_until} every={dens_every} max_K={dens_maxK}")
    logger.info(f"K={field.num_gaussians}")

    # -- timing -----------------------------------------------------------
    timings: dict[str, list[float]] = {
        "sample": [], "vol_fwd": [], "mip_fwd": [], "backward": [], "optim": [],
    }

    # -- early stopping --------------------------------------------------
    es_on = bool(tc.get("early_stopping", False))
    es_patience = int(tc.get("early_stopping_patience", 20))
    es_min_delta = float(tc.get("early_stopping_min_delta", 0.01))  # dB
    es_best_psnr = -float('inf')
    es_no_improve = 0
    es_best_path = None
    if es_on:
        logger.info(f"Early stopping: patience={es_patience} evals, min_delta={es_min_delta} dB")

    best_total = float('inf')
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        # --- Step-0 diagnostic: verify initialization isn't dead ---
        if step == 0:
            with torch.no_grad():
                test_pts, test_vals = sample_points_from_volume(
                    vol_gpu, min(1024, vol_pts),
                    intensity_weighted=iwt, cache_key="vol" if iwt else None,
                )
                test_pred = field(test_pts)
                pred_mean = float(test_pred.mean())
                pred_max = float(test_pred.max())
                pred_nonzero = float((test_pred > 1e-6).float().mean())
                gt_mean = float(test_vals.mean())
                scales_now = torch.exp(field.log_scales)
                amps_now = torch.exp(field.log_amplitudes)
                logger.info(
                    f"INIT CHECK: pred mean={pred_mean:.6f} max={pred_max:.6f} "
                    f"nonzero={pred_nonzero:.1%} | gt mean={gt_mean:.6f} | "
                    f"scale mean={float(scales_now.mean()):.6f} "
                    f"amp mean={float(amps_now.mean()):.6f}"
                )
                if pred_mean < 1e-6:
                    logger.warning(
                        "⚠ Model output is near-zero at init! "
                        "Gaussians are too small or amplitudes too low. "
                        "Increase init_scale or init_amplitude."
                    )

        t_frac = step / max(1, steps - 1)
        tau = compute_tau_schedule(tau_s, tau_e, t_frac)
        wv, wm = weight_schedule(cfg, step, steps)

        # restore LR after densify warmup (let cosine scheduler take over)
        if step - last_dens == dens_lr_warm:
            # Recreate scheduler at correct position with base LR
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=lr * lr_min_frac,
                last_epoch=step,
            )

        optimizer.zero_grad(set_to_none=True)

        # current mode (progressive switches volume → hybrid)
        cur = mode
        if mode == "progressive":
            cur = "volume" if t_frac < prog_split else "hybrid"

        total_loss = torch.zeros((), device=device)
        losses: dict[str, float] = {"tau": tau, "wv": wv, "wm": wm}

        amp_ctx = torch.amp.autocast("cuda") if use_amp else torch.nullcontext()
        with amp_ctx:
            # ---------- volume branch ----------
            if cur in ("volume", "hybrid"):
                t0 = time.time()
                if use_grad:
                    (
                        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                    ) = sample_points_with_neighbors(
                        vol_gpu, vol_pts, delta_vox=delta_vox,
                        intensity_weighted=iwt,
                        cache_key="vol" if iwt else None,
                    )
                else:
                    x, v = sample_points_from_volume(
                        vol_gpu, vol_pts, intensity_weighted=iwt,
                        cache_key="vol" if iwt else None,
                    )
                    x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None

                if device == "cuda":
                    torch.cuda.synchronize()
                timings["sample"].append(time.time() - t0)

                t0 = time.time()
                lv, pv = loss_volume(
                    field, x, v,
                    x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                    w_grad=w_grad if use_grad else 0.0,
                    w_tube=w_tube,
                    w_cross=w_cross,
                    w_scale=w_scale,
                    scale_target=scale_target,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                timings["vol_fwd"].append(time.time() - t0)

                total_loss = total_loss + wv * lv
                for k, vv in pv.items():
                    losses[f"v_{k}"] = float(vv.detach())

            # ---------- MIP branch ----------
            if cur in ("mip", "hybrid"):
                t0 = time.time()
                xy, mt = sample_pixels_from_mip(mip_img, mip_px)
                xy, mt = xy.to(device), mt.to(device)

                lm, pm = loss_mip(
                    field, xy, mt, n_z, tau,
                    w_tube=w_tube, w_cross=w_cross,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                timings["mip_fwd"].append(time.time() - t0)

                total_loss = total_loss + wm * lm
                for k, vv in pm.items():
                    losses[f"m_{k}"] = float(vv.detach())

        # ---------- backward + step ----------
        t0 = time.time()
        if use_amp:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(field.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(field.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        if device == "cuda":
            torch.cuda.synchronize()
        timings["backward"].append(time.time() - t0)

        # ---------- post-step constraints ----------
        if aabb_hard:
            field.apply_aabb_clamp()
        if do_clamp:
            field.clamp_log_scales_(ls_min, ls_max)
        if clamp_amp:
            field.clamp_log_amplitudes_(la_min, la_max)

        losses["total"] = float(total_loss.detach())

        # ---------- accumulate gradient magnitudes for densify ----------
        if dens_on and dens_from <= step <= dens_until:
            g = _gradient_magnitudes(field.means)
            # Handle size mismatch after densify (grad_accum was reset)
            if g.shape[0] == grad_accum.shape[0]:
                grad_accum += g
                grad_count += 1
            else:
                grad_accum = g.clone()
                grad_count = 1

        # ---------- densify / prune ----------
        if (
            dens_on
            and dens_from <= step <= dens_until
            and step % dens_every == 0
            and step > 0
            and grad_count > 0
        ):
            # Use average gradient over accumulation window
            avg_grad = grad_accum / max(grad_count, 1)

            # Log gradient stats for diagnostics
            gmax = float(avg_grad.max())
            gmean = float(avg_grad.mean())
            gmed = float(avg_grad.median())
            above = int((avg_grad > dens_grad).sum())
            logger.info(
                f"GradStats@{step}: max={gmax:.6f} mean={gmean:.6f} "
                f"median={gmed:.6f} above_thresh={above}/{field.num_gaussians} "
                f"(thresh={dens_grad:.6f})"
            )

            # Temporarily inject averaged gradients for densify decision
            old_grad = field.means.grad
            field.means.grad = avg_grad.unsqueeze(-1).expand_as(field.means).contiguous()

            stats = field.densify_and_prune(
                grad_threshold=dens_grad,
                min_opacity=dens_minop,
                max_scale=dens_maxsc,
                split_scale_threshold=dens_split,
                enforce_aabb=dens_aabb,
                max_gaussians=dens_maxK,
                max_clones=dens_max_clones,
            )
            logger.info(f"Densify@{step}: {stats}")

            # Restore original grad (will be None on new params anyway)
            if old_grad is not None and old_grad.shape == field.means.shape:
                field.means.grad = old_grad

            # Reset accumulator for new Gaussian set
            grad_accum = torch.zeros(field.num_gaussians, device=device)
            grad_count = 0

            last_dens = step
            optimizer = torch.optim.Adam(
                field.parameters(), lr=lr * dens_lr_fac
            )
            # Recreate scheduler at current position so cosine decay continues
            for pg in optimizer.param_groups:
                pg.setdefault('initial_lr', lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=steps, eta_min=lr * lr_min_frac,
                last_epoch=step,
            )
            if use_amp:
                scaler = torch.amp.GradScaler("cuda")
            if device == "cuda":
                torch.cuda.empty_cache()

        # ---------- progress bar ----------
        if "total" in losses:
            best_total = min(best_total, losses["total"])
        d = {"total": f"{losses.get('total', 0):.4g}", "best": f"{best_total:.4g}"}
        show_keys = ["v_rec", "v_grad", "v_scale", "m_mip", "wm", "tau"]
        d.update({k: f"{losses[k]:.4g}" for k in show_keys if k in losses})
        pbar.set_postfix(d)

        log_every = int(tc.get("log_every", 50))
        if (step + 1) % log_every == 0:
            logger.info(f"Step {step+1}: K={field.num_gaussians}  best_total={best_total:.6g}  {losses}")

        # ---------- PSNR evaluation ----------
        psnr_every = int(tc.get("psnr_eval_every", 500))
        if (step + 1) % psnr_every == 0:
            with torch.no_grad():
                # Sample 100K random voxels (uniform, not intensity-weighted) for unbiased PSNR
                eval_n = min(100_000, vol_gpu.numel())
                Z, Y, X = vol_gpu.shape
                idx = torch.randint(0, vol_gpu.numel(), (eval_n,), device=device)
                ez = idx // (Y * X)
                ey = (idx % (Y * X)) // X
                ex = idx % X
                exn = (ex.float() / max(X - 1, 1)) * 2 - 1
                eyn = (ey.float() / max(Y - 1, 1)) * 2 - 1
                ezn = (ez.float() / max(Z - 1, 1)) * 2 - 1
                eval_pts = torch.stack([exn, eyn, ezn], dim=1)
                eval_gt = vol_gpu[ez, ey, ex]
                # Chunked evaluation to avoid OOM
                eval_pred_chunks = []
                chunk_sz = 16384
                for ci in range(0, eval_n, chunk_sz):
                    eval_pred_chunks.append(field(eval_pts[ci:ci+chunk_sz]))
                eval_pred = torch.cat(eval_pred_chunks)
                mse = F.mse_loss(eval_pred, eval_gt).item()
                mae = F.l1_loss(eval_pred, eval_gt).item()
                psnr = -10 * math.log10(max(mse, 1e-12))
                logger.info(f"PSNR@{step+1}: {psnr:.2f} dB  (MSE={mse:.6f}, MAE={mae:.6f}, eval_pts={eval_n})")

                # Early stopping check (skip during cooldown after densify)
                steps_since_dens = step - last_dens
                evals_since_dens = steps_since_dens // psnr_every
                in_cooldown = dens_on and (evals_since_dens < dens_cooldown)
                if es_on and in_cooldown:
                    logger.info(f"Early stopping: cooldown ({evals_since_dens+1}/{dens_cooldown} evals after densify)")
                elif es_on:
                    if psnr > es_best_psnr + es_min_delta:
                        es_best_psnr = psnr
                        es_no_improve = 0
                        # Save best checkpoint
                        if save_path:
                            es_best_path = save_path.replace(".pt", "_best.pt")
                            os.makedirs(os.path.dirname(es_best_path) or ".", exist_ok=True)
                            torch.save(field.state_dict(), es_best_path)
                            logger.info(f"New best PSNR: {es_best_psnr:.2f} dB → {es_best_path}")
                    else:
                        es_no_improve += 1
                        logger.info(f"Early stopping: no improvement {es_no_improve}/{es_patience} (best={es_best_psnr:.2f} dB)")
                    if es_no_improve >= es_patience:
                        logger.info(f"Early stopping triggered at step {step+1} (best PSNR: {es_best_psnr:.2f} dB)")
                        break

        # ---------- checkpoint ----------
        save_path = tc.get("save_path")
        ckpt_every = int(tc.get("checkpoint_interval", 1000))
        if save_path and (step + 1) % ckpt_every == 0:
            base = save_path.replace(".pt", "")
            ckpt = f"{base}_step{step+1}.pt"
            os.makedirs(os.path.dirname(ckpt) or ".", exist_ok=True)
            torch.save(field.state_dict(), ckpt)
            logger.info(f"Checkpoint → {ckpt}")

    # -- timing report ----------------------------------------------------
    # -- load best early-stopping checkpoint if available ---------------
    if es_on and es_best_path and os.path.exists(es_best_path):
        field.load_state_dict(torch.load(es_best_path, weights_only=True))
        logger.info(f"Loaded best checkpoint (PSNR {es_best_psnr:.2f} dB) from {es_best_path}")

    logger.info("=" * 60)
    logger.info("TIMING  (mean ± std  ms)")
    logger.info("=" * 60)
    total_ms = 0.0
    for key in ["sample", "vol_fwd", "mip_fwd", "backward"]:
        if timings[key]:
            arr = np.array(timings[key]) * 1000
            total_ms += arr.mean()
            logger.info(f"  {key:12s}: {arr.mean():7.2f} ± {arr.std():5.2f}")
    if total_ms > 0:
        logger.info(f"  {'TOTAL':12s}: {total_ms:7.2f} ms/step  ({1000/total_ms:.1f} it/s)")
    logger.info("=" * 60)
    logger.info("Training finished.")
    return field


# ===================================================================
#  Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Gaussian Mixture Field")
    parser.add_argument("--config", default="config.yml", help="YAML config path")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 0))
    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Seed: {seed}")

    dev = cfg["training"].get("device", "auto")
    device = "cuda" if (dev == "auto" and torch.cuda.is_available()) else dev
    if device != "cuda":
        raise RuntimeError("CUDA required.  Set device: cuda in config.")

    vol = load_tif_data(cfg["data"]["tif_path"])

    mc = cfg["model"]

    # Load SWC morphology for Gaussian initialization (if provided)
    swc_coords, swc_radii = None, None
    swc_path = cfg["data"].get("swc_path")
    if swc_path and os.path.exists(swc_path):
        swc_data = load_swc(swc_path)
        swc_coords, swc_radii = swc_to_normalised_coords(
            swc_data, vol.shape, bounds=mc.get("bounds")
        )
        print(f"Loaded SWC: {swc_data.shape[0]} nodes from {swc_path}")
    elif swc_path:
        print(f"WARNING: swc_path={swc_path} not found, using random init")

    field = GaussianMixtureField(
        num_gaussians=int(mc["num_gaussians"]),
        init_scale=float(mc.get("init_scale", 0.05)),
        init_amplitude=float(mc.get("init_amplitude", 0.1)),
        bounds=mc.get("bounds"),
        aabb=mc.get("aabb"),
        swc_coords=swc_coords,
        swc_radii=swc_radii,
    )

    # Resume from checkpoint if specified
    resume_path = args.resume or cfg["training"].get("resume_from")
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        # Handle checkpoint with different num_gaussians
        K_ckpt = ckpt["means"].shape[0]
        if K_ckpt != field.num_gaussians:
            print(f"Checkpoint has K={K_ckpt} (config has {field.num_gaussians}), adjusting...")
            field = GaussianMixtureField(
                num_gaussians=K_ckpt,
                init_scale=float(mc.get("init_scale", 0.05)),
                init_amplitude=float(mc.get("init_amplitude", 0.1)),
                bounds=mc.get("bounds"),
                aabb=mc.get("aabb"),
            )
        field.load_state_dict(ckpt)
        print(f"Resumed from {resume_path} (K={field.num_gaussians})")
    elif resume_path:
        print(f"WARNING: resume_from={resume_path} not found, training from scratch")

    log_dir = cfg["training"].get("log_dir", "logs")
    field = train(field, vol, cfg, device=device, log_dir=log_dir)

    out = cfg["training"].get("save_path")
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save(field.state_dict(), out)
        print(f"Saved → {out}")


if __name__ == "__main__":
    main()
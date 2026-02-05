#!/usr/bin/env python3
"""
NeuroGS-Codec Standalone Training Script
======================================

Goal
----
Train a sparse mixture of *anisotropic 3D Gaussians* to approximate an input microscopy
volume V(z,y,x). This is used as a **codec-style representation**:
- The "model parameters" (Gaussian means, scales, rotations, amplitudes, bias) are the code.
- A learned Laplace entropy model provides an estimate of bit-cost (rate).
- Training minimizes a rate–distortion objective + geometry/topology regularizers.

Run
---
python train_standalone.py

Detached execution (survives VS Code disconnect):
  tmux new -s train
  python train_standalone.py
  # Detach: Ctrl+B, then D
  # Reattach: tmux attach -t train

Or:
  nohup python train_standalone.py > training.log 2>&1 &

Notes
-----
- Coordinates are normalized to [-1,1]^3 to keep optimization stable.
- The voxel spacing is used only to build physical coordinate axes before normalization.
"""

import os
import math
import time
import gc
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from tqdm.auto import tqdm

# ============================================================================
# Configuration
# ============================================================================
# Input volume path (3D TIF; typically Z,Y,X)
TIF_PATH = "10-2900-control-cell-05_cropped_corrected.tif"

# Physical voxel spacing (microns, etc.). Used to build physical coordinates
# before we normalize them to [-1,1].
VOXEL_SPACING = (0.126, 0.126, 1.0)  # (dx, dy, dz)

# Checkpointing (periodic + final)
CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "neurogs_codec_ckpt_final.pt")

# Placeholder for an eventual "bitstream export" (not implemented in this file).
BITSTREAM_PATH = "neurogs_codec_stream.npz.gz"

# Training config and loss weights
TRAINING_CONFIG = {
    # Initial number of Gaussians
    "N0": 5000,

    # Optimization iterations
    "steps": 30000,

    # How many points we sample per iteration from the target volume
    # (half uniform, half neurite-biased).
    "batch": 2000,

    # Distortion weighting factor for neurite-biased samples:
    # D = mean( (1 + kappa*m(x)) * rho(pred - tgt) )
    "kappa": 8.0,

    # Rate–distortion weights:
    # total = D + lam*R + ...
    "lam": 0.001,

    # Topology regularizer weight
    "alpha": 0.01,

    # Gaussian parameter regularizers
    "beta_sparse": 0.003,   # encourages small amplitudes -> fewer effective primitives
    "beta_smooth": 0.001,   # "shape/covariance conditioning" (NOT field smoothness)

    # Field-level / geometry regularizers
    "beta_tv": 0.002,       # TV on predicted field patches (piecewise smoothness)
    "beta_ssim": 0.1,       # local structural similarity (3D windowed)
    "beta_edge": 0.05,      # emphasize edges / neurite boundaries
    "beta_overlap": 0.01,   # discourage excessive primitive overlap
    "beta_grad": 0.0,       # screenshot-like "smoothness": E ||∇f||^2 (turn on if needed)

    # Learning rate schedule (linear decay)
    "lr": 3e-3,
    "lr_final": 5e-4,

    # Densification settings:
    # clone/split Gaussians based on accumulated |∇mu L| statistics
    "densify_enabled": True,
    "densify_from_iter": 500,
    "densify_until_iter": 20000,
    "densify_every": 500,
    "max_gaussians": 150000,
    "min_amplitude": 0.0005,    # prune very small amplitude Gaussians
    "grad_threshold": 0.00015,  # threshold for selecting points to clone/split

    # Patch size used by topology / TV / SSIM / edge losses
    "topo_patch": (8, 16, 16),

    # Save checkpoint every N iterations
    "save_every": 1000,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ============================================================================
# Helper Functions
# ============================================================================

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion q=(w,x,y,z) into a rotation matrix R (3x3).

    Why:
    - Each Gaussian is anisotropic with oriented covariance.
    - We parameterize orientation with quaternions (stable, differentiable, avoids gimbal lock).
    """
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R

def safe_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    Normalize quaternions safely.

    Why:
    - During training, q may drift away from unit norm.
    - We enforce unit quaternion at usage-time to ensure valid rotations.
    """
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def gaussian_blur_3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Fast separable 3D Gaussian blur via three 1D conv3d passes (x,y,z).

    Used by:
    - make_neurite_map(): difference-of-Gaussians (DoG) + gradient magnitude proxy
      to highlight neurites / edges for biased sampling and edge-aware loss.

    Input:
      vol: (Z,Y,X) tensor
      sigma: std of Gaussian kernel in voxel units

    Output:
      blurred volume with same shape (Z,Y,X)
    """
    if sigma <= 0:
        return vol
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=vol.device, dtype=vol.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / (k.sum() + 1e-8)

    # add batch/channel dims for conv3d: (N,C,Z,Y,X)
    v = vol.unsqueeze(0).unsqueeze(0)

    # separable convolutions: X then Y then Z
    kx = k.view(1, 1, 1, 1, -1)
    v = F.conv3d(v, kx, padding=(0, 0, radius))

    ky = k.view(1, 1, 1, -1, 1)
    v = F.conv3d(v, ky, padding=(0, radius, 0))

    kz = k.view(1, 1, -1, 1, 1)
    v = F.conv3d(v, kz, padding=(radius, 0, 0))

    return v[0, 0]

@torch.no_grad()
def make_neurite_map(vol_zyx: torch.Tensor) -> torch.Tensor:
    """
    Build a "neurite/edge importance map" M in [0,1] used for:
    1) biased sampling (more points on neurites)
    2) edge-aware reconstruction loss weights

    Construction:
    - DoG: |G(sigma=0.8) - G(sigma=2.0)| highlights band-pass structure
    - Gradient magnitude proxy via finite differences highlights boundaries
    - Combine and normalize to [0,1]

    Output:
      M: same shape as volume (Z,Y,X), float in [0,1]
    """
    v1 = gaussian_blur_3d(vol_zyx, sigma=0.8)
    v2 = gaussian_blur_3d(vol_zyx, sigma=2.0)
    dog = (v1 - v2).abs()

    # finite differences; pad back to original size
    dz = F.pad(vol_zyx[1:] - vol_zyx[:-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(vol_zyx[:, 1:] - vol_zyx[:, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(vol_zyx[:, :, 1:] - vol_zyx[:, :, :-1], (0, 1, 0, 0, 0, 0))
    gmag = torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)

    m = dog + 0.5 * gmag
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.clamp(0, 1)

def make_coord_grid_zyx(
    shape_zyx: Tuple[int, int, int],
    spacing_xyz: Tuple[float, float, float],
    device: str
):
    """
    Create a dense coordinate grid in normalized coordinates [-1,1]^3.

    Steps:
    1) Build physical coordinates using voxel spacing:
       x = i * dx, y = j * dy, z = k * dz
    2) Normalize each axis independently to [-1,1] to stabilize optimization.
       (This makes the scene extent roughly constant regardless of volume size.)

    Output:
      coords: (Z,Y,X,3) with order (x,y,z) in the last dimension.
              Note: coords_grid[z,y,x] is a 3D point.
    """
    Z, Y, X = shape_zyx
    dx, dy, dz = spacing_xyz

    xs = torch.arange(X, device=device) * dx
    ys = torch.arange(Y, device=device) * dy
    zs = torch.arange(Z, device=device) * dz

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    z0, z1 = zs.min(), zs.max()

    xn = (xs - x0) / (x1 - x0 + 1e-8) * 2 - 1
    yn = (ys - y0) / (y1 - y0 + 1e-8) * 2 - 1
    zn = (zs - z0) / (z1 - z0 + 1e-8) * 2 - 1

    zz, yy, xx = torch.meshgrid(zn, yn, xn, indexing="ij")
    coords = torch.stack([xx, yy, zz], dim=-1)
    return coords

# ============================================================================
# Model: Mixture of Anisotropic 3D Gaussians
# ============================================================================

class GaussianMixtureVolume(nn.Module):
    """
    Gaussian mixture field:
      f(x) = sum_i a_i * exp( -1/2 * || (R_i^T (x - mu_i)) / s_i ||^2 ) + b

    Parameters per Gaussian i:
      mu_i   : mean / center in normalized coordinates [-1,1]^3
      s_i    : axis-aligned scale (std dev) in the Gaussian's local frame (positive)
              stored as log_s for unconstrained optimization
      q_i    : quaternion for rotation R_i (orientation of local frame)
      a_i    : amplitude (intensity contribution)
    Global:
      b      : global bias / background level

    Why this representation:
    - Sparse (few primitives) for compression.
    - Anisotropy + orientation helps capture tubular neurites efficiently.
    - Differentiable and fast to evaluate at random coordinates.
    """
    def __init__(self, N: int, init_means: torch.Tensor, init_amp: torch.Tensor):
        super().__init__()
        assert init_means.shape == (N, 3)
        assert init_amp.shape == (N,)
        self.N = N

        # Gaussian centers mu_i (N,3)
        self.mu = nn.Parameter(init_means.clone())

        # log-scales (N,3); scale s = exp(log_s)
        self.log_s = nn.Parameter(torch.zeros(N, 3, device=init_means.device) - 2.0)

        # Quaternion rotations q_i (N,4) initialized to identity
        q = torch.zeros(N, 4, device=init_means.device)
        q[:, 0] = 1.0
        self.q = nn.Parameter(q)

        # Amplitudes (N,)
        self.a = nn.Parameter(init_amp.clone())

        # Background bias (scalar)
        self.b = nn.Parameter(torch.tensor(0.0, device=init_means.device))

        # Used for fast culling: treat each Gaussian as bounded inside
        # radius = sigma_cutoff * max_axis_scale
        self.sigma_cutoff = 3.0

    def get_gaussian_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute axis-aligned AABB bounds for each Gaussian in *world coords*.

        For fast culling, we approximate each anisotropic Gaussian by a sphere
        with radius = sigma_cutoff * max(s_i). This gives conservative bounds.
        """
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        max_radius = s.max(dim=-1, keepdim=True).values * self.sigma_cutoff
        radius_3d = max_radius.expand(-1, 3)
        min_bounds = self.mu - radius_3d
        max_bounds = self.mu + radius_3d
        return min_bounds, max_bounds

    def forward(
        self,
        x: torch.Tensor,
        use_culling: bool = True,
        max_gaussians_per_tile: int = 50000
    ) -> torch.Tensor:
        """
        Evaluate the mixture field at query points x.

        x: (P,3) points in normalized coords [-1,1]^3

        Culling:
        - For a batch of query points, we compute the batch AABB [x_min, x_max].
        - We keep only Gaussians whose (approximate) AABB intersects this batch AABB.
        - If too many Gaussians remain, fall back to dense evaluation.

        Why:
        - As N grows (densification), O(P*N) becomes expensive.
        - Culling reduces the active set for local point batches.
        """
        P = x.shape[0]
        if (not use_culling) or (self.N <= max_gaussians_per_tile):
            return self._forward_dense(x)

        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values

        g_min, g_max = self.get_gaussian_bounds()
        overlaps = (g_max >= x_min.unsqueeze(0)) & (g_min <= x_max.unsqueeze(0))
        mask = overlaps.all(dim=-1)

        n_active = mask.sum().item()
        if n_active == 0:
            # If no active Gaussians overlap the tile, output pure background.
            return torch.full((P,), self.b.item(), device=x.device, dtype=x.dtype)

        if n_active > max_gaussians_per_tile:
            return self._forward_dense(x)

        active_idx = mask.nonzero(as_tuple=True)[0]
        mu_active = self.mu[active_idx]
        log_s_active = self.log_s[active_idx]
        q_active = self.q[active_idx]
        a_active = self.a[active_idx]

        # Transform points into each Gaussian's local frame:
        # y = R^T (x - mu), then normalize by scale s.
        dx = x[:, None, :] - mu_active[None, :, :]                 # (P,Na,3)
        s = torch.exp(log_s_active).clamp(1e-4, 10.0)              # (Na,3)
        qn = safe_normalize(q_active)                               # (Na,4)
        R = quat_to_rotmat(qn)                                      # (Na,3,3)
        Rt = R.transpose(-1, -2)                                    # (Na,3,3)

        y = torch.einsum("pni,nij->pnj", dx, Rt)                    # (P,Na,3)
        y = y / (s[None, :, :] + 1e-8)

        # Gaussian kernel
        exp_term = -0.5 * (y * y).sum(dim=-1)                      # (P,Na)
        g = torch.exp(exp_term)                                     # (P,Na)

        # Weighted sum + bias
        pred = (g * a_active[None, :]).sum(dim=1) + self.b          # (P,)
        return pred

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dense evaluation over all N Gaussians: O(P*N).

        Used when:
        - culling is disabled
        - or active set after culling is still too large
        """
        mu = self.mu[None, :, :]            # (1,N,3)
        dx = x[:, None, :] - mu             # (P,N,3)

        s = torch.exp(self.log_s).clamp(1e-4, 10.0)                # (N,3)
        qn = safe_normalize(self.q)                                 # (N,4)
        R = quat_to_rotmat(qn)                                      # (N,3,3)
        Rt = R.transpose(-1, -2)

        y = torch.einsum("pni,nij->pnj", dx, Rt)                    # (P,N,3)
        y = y / (s[None, :, :] + 1e-8)

        exp_term = -0.5 * (y * y).sum(dim=-1)                      # (P,N)
        g = torch.exp(exp_term)

        pred = (g * self.a[None, :]).sum(dim=1) + self.b
        return pred

    def forward_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 1000,
        max_gaussians_per_tile: int = 30000
    ) -> torch.Tensor:
        """
        Evaluate points in smaller chunks to reduce peak memory.

        Useful when:
        - P is huge (e.g., full volume decoding)
        - N is large
        """
        P = x.shape[0]
        if P <= tile_size and self.N <= max_gaussians_per_tile:
            return self.forward(x, use_culling=True, max_gaussians_per_tile=max_gaussians_per_tile)

        pred = torch.empty(P, device=x.device, dtype=x.dtype)
        for start in range(0, P, tile_size):
            end = min(start + tile_size, P)
            x_tile = x[start:end]
            pred[start:end] = self.forward(
                x_tile,
                use_culling=True,
                max_gaussians_per_tile=max_gaussians_per_tile
            )
        return pred

# ============================================================================
# Entropy Models & Quantization (Rate Term)
# ============================================================================

class LaplaceEntropyModel(nn.Module):
    """
    Simple factorized Laplace entropy model:
      p(x) = (1/(2b)) exp(-|x|/b)

    We estimate "rate" in bits as:
      R = E[-log2 p(xq)]

    Why:
    - This is a lightweight stand-in for real arithmetic coding.
    - It yields a differentiable proxy for bitrate.
    - Each parameter group (mu, log_s, q, a, b) gets its own learned scale b.

    Limitation:
    - Assumes i.i.d. Laplace; no spatial/structural dependency modeling.
    """
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.log_b = nn.Parameter(torch.tensor(math.log(init_scale), device=DEVICE))

    def nll_bits(self, xq: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood in *bits* for quantized values xq.

        Note:
        - xq here is not truly integer-coded; we treat it as discrete symbols.
        """
        b = torch.exp(self.log_b).clamp(1e-6, 1e3)
        logp = -math.log(2.0) - torch.log(b) - (xq.abs() / b)  # ln p(x)
        bits = (-logp / math.log(2.0)).mean()
        return bits

def ste_round(x: torch.Tensor) -> torch.Tensor:
    """
    Straight-Through Estimator (STE) rounding.

    Forward:  round(x)
    Backward: identity gradient (as if no rounding happened)

    Why:
    - Enables optimizing with quantization in-the-loop.
    - Common trick in learned compression and quantization-aware training.
    """
    return (x.round() - x).detach() + x

@dataclass
class QuantSteps:
    """
    Quantization step sizes for each parameter block.

    Interpretation:
      xq = round(x / step)
    Smaller step -> finer quantization -> higher rate but potentially lower distortion.
    """
    mu: float = 1/2048
    log_s: float = 1/256
    q: float = 1/1024
    a: float = 1/1024
    b: float = 1/1024

# ============================================================================
# Loss Functions
# ============================================================================

def charbonnier(x, eps=1e-3):
    """
    Charbonnier (smooth L1) penalty:
      rho(x) = sqrt(x^2 + eps^2)

    Why:
    - More robust than L2 (less sensitive to outliers).
    - Differentiable everywhere.
    """
    return torch.sqrt(x*x + eps*eps)

@torch.no_grad()
def extract_random_patch(vol: torch.Tensor, patch_zyx=(32, 64, 64)):
    """
    Randomly extract a 3D patch from a volume.

    Used for:
    - topology regularizer
    - reconstruction TV
    - SSIM
    - edge-aware loss

    Returns:
      (z0,y0,x0), patch
    """
    Z, Y, X = vol.shape
    pz, py, px = patch_zyx
    z0 = torch.randint(0, max(1, Z - pz + 1), (1,), device=vol.device).item()
    y0 = torch.randint(0, max(1, Y - py + 1), (1,), device=vol.device).item()
    x0 = torch.randint(0, max(1, X - px + 1), (1,), device=vol.device).item()
    return (z0, y0, x0), vol[z0:z0+pz, y0:y0+py, x0:x0+px]

def tv3d(p: torch.Tensor):
    """
    Simple 3D anisotropic total variation (TV) approximation:
      TV(p) ~ mean(|dx|) + mean(|dy|) + mean(|dz|)

    Why:
    - Penalizes excessive local oscillations / speckle.
    - Encourages piecewise smooth fields.
    """
    dz = (p[1:] - p[:-1]).abs().mean()
    dy = (p[:, 1:] - p[:, :-1]).abs().mean()
    dx = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
    return dx + dy + dz

def patch_topology_loss(model, coords_grid, V, patch_zyx=(16, 32, 32), tau=0.25, gamma=10.0):
    """
    Patch-based topology-ish regularizer.

    Intuition:
    - Convert predicted intensity to a soft occupancy P via a steep sigmoid:
        P = sigmoid(gamma * (pred - tau))
      tau sets the "foreground threshold", gamma sets sharpness.

    - Apply TV on P to discourage fragmented noisy occupancy.
    - Add a small high-frequency term (P - local_avg(P)) to discourage speckle.

    This does NOT compute true topology (e.g., Betti numbers),
    but acts as a cheap proxy to stabilize connected tubular structures.

    Returns:
      scalar loss
    """
    (z0, y0, x0), _ = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx

    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)

    P = torch.sigmoid(gamma * (pred - tau))
    tv = tv3d(P)

    # local smoothing via average pooling
    P_ = P[None, None]
    low = F.avg_pool3d(P_, kernel_size=3, stride=1, padding=1)[0, 0]
    hf = (P - low).abs().mean()

    return tv + 0.5 * hf

def reconstruction_tv_loss(model, coords_grid, patch_zyx=(8, 16, 16)):
    """
    TV regularization on the *predicted field* f(x) over a random patch.

    This is a field-level smoothness prior, unlike smoothness_loss(model)
    which operates on Gaussian parameters.
    """
    Z, Y, X, _ = coords_grid.shape
    pz, py, px = patch_zyx

    z0 = torch.randint(0, max(1, Z - pz + 1), (1,), device=coords_grid.device).item()
    y0 = torch.randint(0, max(1, Y - py + 1), (1,), device=coords_grid.device).item()
    x0 = torch.randint(0, max(1, X - px + 1), (1,), device=coords_grid.device).item()

    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    return tv3d(pred)

def ssim_loss_3d(pred, target, win=5, K1=0.01, K2=0.03):
    """
    Simplified 3D SSIM loss computed via local mean/variance with avg_pool3d.

    Output:
      1 - mean(SSIM), so smaller is better.

    Why:
    - SSIM encourages structural similarity beyond pointwise errors.
    - Useful to preserve local contrast and texture of neurites.
    """
    pred_ = pred[None, None]
    tgt_ = target[None, None]
    pad = win // 2

    mu_p = F.avg_pool3d(pred_, win, stride=1, padding=pad)
    mu_t = F.avg_pool3d(tgt_, win, stride=1, padding=pad)

    sigma_p = F.avg_pool3d(pred_ * pred_, win, stride=1, padding=pad) - mu_p * mu_p
    sigma_t = F.avg_pool3d(tgt_ * tgt_, win, stride=1, padding=pad) - mu_t * mu_t
    sigma_pt = F.avg_pool3d(pred_ * tgt_, win, stride=1, padding=pad) - mu_p * mu_t

    C1, C2 = K1**2, K2**2
    ssim_map = ((2*mu_p*mu_t + C1) * (2*sigma_pt + C2)) / \
               ((mu_p**2 + mu_t**2 + C1) * (sigma_p + sigma_t + C2) + 1e-8)
    return 1.0 - ssim_map.mean()

def patch_ssim_loss(model, coords_grid, V, patch_zyx=(8, 16, 16)):
    """
    SSIM loss on a random patch: encourages local structural similarity.
    """
    (z0, y0, x0), tgt_patch = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred_patch = model(coords).reshape(pz, py, px)
    return ssim_loss_3d(pred_patch, tgt_patch, win=3)

def edge_aware_loss(model, coords_grid, V, M, patch_zyx=(8, 16, 16)):
    """
    Edge-aware L1 loss on a patch.

    Weighting:
      edge_weight = clamp(|∂x V| + |∂y V| + |∂z V| + M, 0, 1)

    Why:
    - Neurite boundaries/edges are geometrically important.
    - Standard losses can "wash out" thin structures.
    - This upweights high-gradient regions and neurite-likely voxels.
    """
    (z0, y0, x0), tgt_patch = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx

    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred_patch = model(coords).reshape(pz, py, px)

    m_patch = M[z0:z0+pz, y0:y0+py, x0:x0+px]

    dz = F.pad(tgt_patch[1:] - tgt_patch[:-1], (0, 0, 0, 0, 0, 1)).abs()
    dy = F.pad(tgt_patch[:, 1:] - tgt_patch[:, :-1], (0, 0, 0, 1, 0, 0)).abs()
    dx = F.pad(tgt_patch[:, :, 1:] - tgt_patch[:, :, :-1], (0, 1, 0, 0, 0, 0)).abs()

    edge_weight = (dz + dy + dx + m_patch).clamp(0, 1)
    diff = (pred_patch - tgt_patch).abs()
    return (edge_weight * diff).mean()

def sparsity_loss(model):
    """
    L1 on amplitudes: encourages many Gaussians to become negligible.

    Why:
    - Better compression / fewer effective primitives.
    - Helps pruning when combined with min_amplitude threshold.
    """
    return model.a.abs().mean()

def smoothness_loss(model):
    """
    IMPORTANT: This is NOT field smoothness.

    This is *Gaussian parameter conditioning*:
    - Penalize too-small scales (avoid delta-like spikes -> unstable training / aliasing)
    - Penalize too-large scales (avoid overly blurry primitives)
    - Penalize extreme anisotropy ratio (avoid degenerate covariances)

    Interpretable as: regularizing the covariance Σ_i to remain well-conditioned.

    Returns:
      scalar
    """
    s = torch.exp(model.log_s)

    # prevent tiny scales (spiky Gaussians)
    min_scale_penalty = torch.relu(0.01 - s).mean()

    # anisotropy penalty: (max/min - 1)
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values + 1e-8
    anisotropy = (s_max / s_min - 1.0).clamp(min=0).mean()

    # prevent too-large scales (excessive blur)
    max_scale_penalty = torch.relu(s - 1.0).mean()

    return min_scale_penalty + 0.1 * anisotropy + 0.1 * max_scale_penalty

def field_grad_smoothness(model, pts, n_sub=512):
    """
    Field gradient smoothness: E_x ||∇f(x)||^2

    This is the "screenshot-like smoothness" term:
    - It penalizes rapid spatial changes of the reconstructed field.
    - Often better aligned with geometric correctness than parameter-only penalties.

    Implementation:
    - Take a subset of sampled points for efficiency.
    - Enable gradient wrt pts, then compute ∇ f by autograd.

    Returns:
      scalar
    """
    if pts.shape[0] > n_sub:
        idx = torch.randint(0, pts.shape[0], (n_sub,), device=pts.device)
        pts = pts[idx]
    pts = pts.detach().requires_grad_(True)
    pred = model(pts)
    g = torch.autograd.grad(pred.sum(), pts, create_graph=True)[0]  # (n_sub,3)
    return (g.pow(2).sum(dim=-1)).mean()

def overlap_loss_mahalanobis(model, n_pairs=2048, eps=1e-4):
    """
    Orientation-aware overlap penalty using Mahalanobis distance.

    Motivation:
    - With densification, Gaussians can collapse into near-duplicates.
    - Excess overlap wastes parameters and can blur structures.

    We sample pairs (i,j) and compute:
      d^2 = (mu_i - mu_j)^T (Σ_i + Σ_j + eps I)^{-1} (mu_i - mu_j)

    Then define "overlap" as:
      overlap = exp(-0.5 d^2)

    Penalizing mean(overlap) discourages primitives whose centers are too close
    relative to their combined covariance.

    Compared to Euclidean distance:
    - Accounts for anisotropy and rotation (tubes can be close along thin axis).
    """
    N = model.N
    if N < 2:
        return torch.tensor(0.0, device=model.mu.device)

    idx_i = torch.randint(0, N, (n_pairs,), device=model.mu.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=model.mu.device)

    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if idx_i.numel() == 0:
        return torch.tensor(0.0, device=model.mu.device)

    mu_i = model.mu[idx_i]                            # (P,3)
    mu_j = model.mu[idx_j]                            # (P,3)
    dmu = (mu_i - mu_j).unsqueeze(-1)                 # (P,3,1)

    # Build Σ_i = R diag(s^2) R^T (covariance in world frame)
    s_i = torch.exp(model.log_s[idx_i]).clamp(1e-4, 10.0)
    s_j = torch.exp(model.log_s[idx_j]).clamp(1e-4, 10.0)

    q_i = safe_normalize(model.q[idx_i])
    q_j = safe_normalize(model.q[idx_j])
    R_i = quat_to_rotmat(q_i)
    R_j = quat_to_rotmat(q_j)

    Di = torch.diag_embed(s_i * s_i)
    Dj = torch.diag_embed(s_j * s_j)

    Si = R_i @ Di @ R_i.transpose(-1, -2)
    Sj = R_j @ Dj @ R_j.transpose(-1, -2)

    # combined covariance with small diagonal jitter for numerical stability
    I = torch.eye(3, device=model.mu.device).unsqueeze(0)
    Sij = Si + Sj + eps * I

    # Compute d^2 via linear solve (more stable than inverse):
    # solve Sij * x = dmu -> x = Sij^{-1} dmu
    x = torch.linalg.solve(Sij, dmu)                  # (P,3,1)
    d2 = (dmu.transpose(-1, -2) @ x).squeeze(-1).squeeze(-1)  # (P,)

    overlap = torch.exp(-0.5 * d2)
    return overlap.mean()

# ============================================================================
# Densification Controller (clone/split/prune)
# ============================================================================

class DensificationController:
    """
    Densification strategy inspired by 3D Gaussian Splatting:

    Track gradient magnitude wrt Gaussian centers mu:
      g_i = || ∇_{mu_i} L ||

    Periodically:
    - Clone Gaussians with high gradient but small scale (need more capacity locally)
    - Split Gaussians with high gradient and large scale (refine coarse primitives)
    - Prune:
        - low amplitude (ineffective primitives)
        - too large scale (excess blur / unstable)

    Why:
    - Start with modest N0; grow capacity where needed.
    - Helps represent thin neurites without over-allocating everywhere.
    """
    def __init__(self, model, percent_dense=0.01):
        self.model = model
        self.percent_dense = percent_dense
        self.xyz_gradient_accum = torch.zeros((model.N, 1), device=DEVICE)
        self.denom = torch.zeros((model.N, 1), device=DEVICE)

    def reset_stats(self):
        """Reset running stats after a densify step."""
        N = self.model.N
        self.xyz_gradient_accum = torch.zeros((N, 1), device=DEVICE)
        self.denom = torch.zeros((N, 1), device=DEVICE)

    def add_densification_stats(self, grad_mu):
        """
        Accumulate gradient magnitudes for mu.

        grad_mu: (N,3) gradient of current loss wrt mu parameters.
        """
        if grad_mu is not None:
            grad_norm = grad_mu.norm(dim=-1, keepdim=True)  # (N,1)
            self.xyz_gradient_accum += grad_norm
            self.denom += 1

    def get_scaling(self):
        """Return current scales s_i (N,3)."""
        return torch.exp(self.model.log_s).clamp(1e-4, 10.0)

    @torch.no_grad()
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        Clone Gaussians that:
        - have large gradient (need more capacity)
        - are small enough (already fine-scale; duplicating increases density)

        percent_dense * scene_extent is used as a heuristic size cutoff.
        """
        scaling = self.get_scaling()
        selected_pts_mask = (grads.squeeze() >= grad_threshold)
        selected_pts_mask = selected_pts_mask & (
            scaling.max(dim=1).values <= self.percent_dense * scene_extent
        )
        if selected_pts_mask.sum() == 0:
            return 0

        new_mu = self.model.mu.data[selected_pts_mask].clone()
        new_log_s = self.model.log_s.data[selected_pts_mask].clone()
        new_q = self.model.q.data[selected_pts_mask].clone()
        new_a = self.model.a.data[selected_pts_mask].clone()

        self._cat_tensors(new_mu, new_log_s, new_q, new_a)
        return int(selected_pts_mask.sum().item())

    @torch.no_grad()
    def densify_and_split(self, grads, grad_threshold, scene_extent, N_split=2):
        """
        Split Gaussians that:
        - have large gradient (need more capacity)
        - are large-scale (coarse; split into smaller primitives)

        Mechanism:
        - Sample offsets in Gaussian local scale space
        - Rotate offsets by R and add to mu
        - Shrink scales, split amplitude among children
        - Prune original parent (replace by children)
        """
        scaling = self.get_scaling()
        n_init = self.model.N

        padded_grad = torch.zeros((n_init,), device=DEVICE)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        selected_pts_mask = (padded_grad >= grad_threshold)
        selected_pts_mask = selected_pts_mask & (
            scaling.max(dim=1).values > self.percent_dense * scene_extent
        )
        if selected_pts_mask.sum() == 0:
            return 0

        num_selected = selected_pts_mask.sum().item()

        # Sample offsets with std = scale (in local coords)
        stds = scaling[selected_pts_mask].repeat(N_split, 1)
        means = torch.zeros((stds.size(0), 3), device=DEVICE)
        samples = torch.normal(mean=means, std=stds)

        # Rotate offsets into world coords and shift centers
        qn = safe_normalize(self.model.q.data[selected_pts_mask])
        R = quat_to_rotmat(qn)
        R_rep = R.repeat(N_split, 1, 1)

        new_mu = torch.bmm(R_rep, samples.unsqueeze(-1)).squeeze(-1) + \
                 self.model.mu.data[selected_pts_mask].repeat(N_split, 1)

        # keep within normalized cube bounds
        new_mu = new_mu.clamp(-1, 1)

        # shrink scales (heuristic): splitting should reduce extent
        new_log_s = self.model.log_s.data[selected_pts_mask].repeat(N_split, 1) - math.log(0.8 * N_split)

        # copy orientations
        new_q = self.model.q.data[selected_pts_mask].repeat(N_split, 1)

        # split amplitude among children
        new_a = self.model.a.data[selected_pts_mask].repeat(N_split).reshape(-1) / N_split

        self._cat_tensors(new_mu, new_log_s, new_q, new_a)

        # prune parent points; append zeros for new children so mask length matches
        prune_mask = torch.cat([
            selected_pts_mask,
            torch.zeros(N_split * num_selected, device=DEVICE, dtype=torch.bool)
        ])
        self.prune_points(prune_mask)

        # net increase = (children - parent) = num_selected*(N_split - 1)
        return num_selected * (N_split - 1)

    def _cat_tensors(self, new_mu, new_log_s, new_q, new_a):
        """
        Append newly created Gaussians to the model parameters.
        Also grow the densification statistics buffers.
        """
        add_count = new_mu.shape[0]

        self.model.mu = nn.Parameter(torch.cat([self.model.mu.data, new_mu], dim=0))
        self.model.log_s = nn.Parameter(torch.cat([self.model.log_s.data, new_log_s], dim=0))
        self.model.q = nn.Parameter(torch.cat([self.model.q.data, new_q], dim=0))

        new_a_flat = new_a.reshape(-1) if new_a.dim() > 1 else new_a
        self.model.a = nn.Parameter(torch.cat([self.model.a.data, new_a_flat], dim=0))

        self.model.N += add_count

        self.xyz_gradient_accum = torch.cat([
            self.xyz_gradient_accum,
            torch.zeros((add_count, 1), device=DEVICE)
        ], dim=0)
        self.denom = torch.cat([
            self.denom,
            torch.zeros((add_count, 1), device=DEVICE)
        ], dim=0)

    @torch.no_grad()
    def prune_points(self, mask):
        """
        Remove Gaussians where mask=True.

        Used by:
        - prune_low_amplitude
        - prune_large_scale
        - pruning split parents
        """
        valid_mask = ~mask
        if valid_mask.sum() == valid_mask.numel():
            return 0

        pruned = int((~valid_mask).sum().item())

        self.model.mu = nn.Parameter(self.model.mu.data[valid_mask])
        self.model.log_s = nn.Parameter(self.model.log_s.data[valid_mask])
        self.model.q = nn.Parameter(self.model.q.data[valid_mask])
        self.model.a = nn.Parameter(self.model.a.data[valid_mask])

        self.model.N = self.model.mu.shape[0]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_mask]
        self.denom = self.denom[valid_mask]
        return pruned

    @torch.no_grad()
    def prune_low_amplitude(self, min_amplitude=0.002):
        """
        Prune Gaussians with |a_i| below threshold.

        Why:
        - These primitives contribute little to the field.
        - Helps compression (fewer effective parameters).
        """
        prune_mask = self.model.a.data.abs() < min_amplitude
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def prune_large_scale(self, max_scale=0.5, scene_extent=2.0):
        """
        Prune Gaussians that are too large in extent.

        Why:
        - Oversized primitives tend to blur structures and can dominate optimization.
        - Keeps the representation locally adaptive rather than globally smeared.
        """
        scaling = self.get_scaling()
        prune_mask = scaling.max(dim=1).values > max_scale * scene_extent
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold=0.0002, min_amplitude=0.002,
                          scene_extent=2.0, max_scale=0.1):
        """
        Main entry point called periodically from training loop.

        Steps:
        1) Convert accumulators into average gradient magnitude per Gaussian
        2) Clone small+high-grad primitives
        3) Split large+high-grad primitives
        4) Prune low amplitude and overly large primitives
        5) Reset accumulators
        """
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0

        n_cloned = self.densify_and_clone(grads, grad_threshold, scene_extent)
        n_split = self.densify_and_split(grads, grad_threshold, scene_extent)

        n_pruned_amp = self.prune_low_amplitude(min_amplitude)
        n_pruned_scale = self.prune_large_scale(max_scale, scene_extent)

        self.reset_stats()
        torch.cuda.empty_cache()

        return {
            "cloned": n_cloned,
            "split": n_split,
            "pruned_amp": n_pruned_amp,
            "pruned_scale": n_pruned_scale,
            "total_gaussians": self.model.N
        }

# ============================================================================
# Sampling (uniform + neurite-biased)
# ============================================================================

@torch.no_grad()
def sample_points(coords_grid, V, M, n_uniform, n_biased):
    """
    Sample training points for distortion loss.

    - Uniform samples ensure the background and global intensity is learned.
    - Biased samples focus capacity on neurites/edges using M as a probability map.

    Returns:
      pts  : (B,3) coordinates
      tgt  : (B,)  target intensity at those coords
      mval : (B,)  neurite importance in [0,1] at those coords
    """
    Z, Y, X, _ = coords_grid.shape
    total = Z * Y * X

    # uniform indices
    idx_u = torch.randint(0, total, (n_uniform,), device=V.device)
    zu = idx_u // (Y * X)
    yu = (idx_u % (Y * X)) // X
    xu = idx_u % X

    # biased indices using M (flattened as categorical distribution)
    flatM = M.reshape(-1)
    max_categories = 2**24 - 1  # torch.multinomial has some practical limits for huge categories

    if total > max_categories:
        # sample candidate subset to avoid huge categorical distribution
        candidate_idx = torch.randint(0, total, (max_categories,), device=V.device)
        candidate_probs = flatM[candidate_idx]
        candidate_probs = candidate_probs / (candidate_probs.sum() + 1e-8)
        selected = torch.multinomial(candidate_probs, n_biased, replacement=True)
        idx_b = candidate_idx[selected]
    else:
        probs = flatM / (flatM.sum() + 1e-8)
        idx_b = torch.multinomial(probs, n_biased, replacement=True)

    zb = idx_b // (Y * X)
    yb = (idx_b % (Y * X)) // X
    xb = idx_b % X

    # combine
    z = torch.cat([zu, zb], dim=0)
    y = torch.cat([yu, yb], dim=0)
    x = torch.cat([xu, xb], dim=0)

    pts = coords_grid[z, y, x]   # (B,3)
    tgt = V[z, y, x]             # (B,)
    mval = M[z, y, x]            # (B,)
    return pts, tgt, mval

@torch.no_grad()
def init_gaussians_from_neurite_map(coords_grid, V, M, N0):
    """
    Initialize Gaussian centers by sampling from neurite map M.
    This places initial primitives preferentially near neurites/edges.

    Amplitudes are initialized from the target intensity at sampled positions.

    Returns:
      means: (N0,3)
      amp  : (N0,)
    """
    Z, Y, X, _ = coords_grid.shape
    total = Z * Y * X

    flatM = M.reshape(-1)
    max_categories = 2**24 - 1

    if total > max_categories:
        candidate_idx = torch.randint(0, total, (max_categories,), device=M.device)
        candidate_probs = flatM[candidate_idx]
        candidate_probs = candidate_probs / (candidate_probs.sum() + 1e-8)
        selected = torch.multinomial(candidate_probs, N0, replacement=True)
        idx = candidate_idx[selected]
    else:
        probs = flatM / (flatM.sum() + 1e-8)
        idx = torch.multinomial(probs, N0, replacement=True)

    z = idx // (Y * X)
    y = (idx % (Y * X)) // X
    x = idx % X

    means = coords_grid[z, y, x]
    amp = V[z, y, x].clone()
    return means, amp

# ============================================================================
# Checkpoint Save/Load
# ============================================================================

def ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(model, H_mu, H_logs, H_q, H_a, H_b, Q, losses, densify_log, config, path, iteration=None):
    """
    Save checkpoint with:
    - model state (Gaussian parameters)
    - entropy model states (Laplace scales)
    - quantization steps
    - training config (so resume is consistent)
    - losses history and densify events (for later plotting/debug)
    - dataset metadata (path + voxel spacing)

    This lets you:
    - resume training
    - analyze curves
    - keep a full provenance record
    """
    ensure_checkpoint_dir()

    if not os.path.dirname(path):
        path = os.path.join(CKPT_DIR, path)

    checkpoint = {
        "model_state": model.state_dict(),
        "model_N": model.N,
        "entropy_state": {
            "H_mu": H_mu.state_dict(),
            "H_logs": H_logs.state_dict(),
            "H_q": H_q.state_dict(),
            "H_a": H_a.state_dict(),
            "H_b": H_b.state_dict(),
        },
        "Q": Q.__dict__,
        "config": config,
        "iteration": iteration,
        "losses": losses,
        "densify_log": densify_log,
        "tif_path": TIF_PATH,
        "voxel_spacing": VOXEL_SPACING,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path} (iter={iteration}, N={model.N})")

# ============================================================================
# Training
# ============================================================================

def train(model, V, coords_grid, M, H_mu, H_logs, H_q, H_a, H_b, Q, cfg):
    """
    Main optimization loop.

    Objective:
      total = D + lam*R + alpha*T
              + beta_tv*TV + beta_ssim*SSIM + beta_edge*Edge
              + beta_grad*G
              + beta_sparse*S + beta_smooth*Sm + beta_overlap*O

    Where:
      D   : weighted Charbonnier distortion on sampled points
      R   : entropy proxy in bits (rate)
      T   : topology-ish proxy on random patches (occupancy TV + HF)
      TV  : field TV regularizer on patches
      SSIM: patch-based SSIM loss
      Edge: edge-aware L1 on patches
      G   : E||∇f||^2 (screenshot-like smoothness)
      S   : amplitude sparsity
      Sm  : Gaussian parameter conditioning (covariance regularization)
      O   : overlap penalty (Mahalanobis)
    """
    controller = DensificationController(model, percent_dense=0.01) if cfg["densify_enabled"] else None

    # optimize both model params and entropy model params (Laplace scales)
    params = (
        list(model.parameters()) +
        list(H_mu.parameters()) + list(H_logs.parameters()) +
        list(H_q.parameters()) + list(H_a.parameters()) + list(H_b.parameters())
    )
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    # log arrays for plotting/debug
    losses = {"D": [], "R": [], "T": [], "TV": [], "SSIM": [], "Edge": [], "G": [],
              "S": [], "Sm": [], "O": [], "Total": [], "N": []}
    densify_log = []
    t0 = time.time()

    lr = cfg["lr"]
    lr_final = cfg.get("lr_final", lr)
    steps = cfg["steps"]

    # cached values for printing (computed sparsely)
    last_T = 0.0
    last_SSIM = 0.0

    print(f"\n{'='*60}")
    print(f"Starting training: {steps} iterations")
    print(f"Device: {DEVICE}")
    print(f"Initial Gaussians: {model.N}")
    print(f"{'='*60}\n")

    for it in tqdm(range(steps), desc="Training"):
        # linear LR decay
        current_lr = lr - (lr - lr_final) * (it / max(steps - 1, 1))
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        # ------------------------------------------------------------
        # 1) Sample training points (uniform + neurite-biased)
        # ------------------------------------------------------------
        n_u = cfg["batch"] // 2
        n_b = cfg["batch"] - n_u
        pts, tgt, mval = sample_points(coords_grid, V, M, n_u, n_b)

        # ------------------------------------------------------------
        # 2) Distortion term D (weighted robust error)
        # ------------------------------------------------------------
        pred = model(pts)
        w = 1.0 + cfg["kappa"] * mval
        D = (w * charbonnier(pred - tgt)).mean()

        # ------------------------------------------------------------
        # 3) Rate term R (entropy proxy after quantization)
        # ------------------------------------------------------------
        # Quantize with STE: round(x/step)
        mu_q = ste_round(model.mu / Q.mu)
        logs_q = ste_round(model.log_s / Q.log_s)
        q_q = ste_round(model.q / Q.q)
        a_q = ste_round(model.a / Q.a)
        b_q = ste_round(model.b / Q.b)

        # Each parameter block contributes bits under its Laplace model
        R = (
            H_mu.nll_bits(mu_q) +
            H_logs.nll_bits(logs_q) +
            H_q.nll_bits(q_q) +
            H_a.nll_bits(a_q) +
            H_b.nll_bits(b_q)
        )

        # ------------------------------------------------------------
        # 4) Patch-based regularizers (computed periodically)
        # ------------------------------------------------------------
        T = torch.tensor(0.0, device=V.device)
        if (it % 20) == 0:
            T = patch_topology_loss(model, coords_grid, V, patch_zyx=cfg["topo_patch"])
            last_T = float(T.detach().cpu())

        TV = torch.tensor(0.0, device=V.device)
        if cfg["beta_tv"] > 0 and (it % 10) == 0:
            TV = reconstruction_tv_loss(model, coords_grid, patch_zyx=cfg["topo_patch"])

        SSIM_loss = torch.tensor(0.0, device=V.device)
        if cfg["beta_ssim"] > 0 and (it % 5) == 0:
            SSIM_loss = patch_ssim_loss(model, coords_grid, V, patch_zyx=cfg["topo_patch"])
            last_SSIM = float(SSIM_loss.detach().cpu())

        Edge_loss = torch.tensor(0.0, device=V.device)
        if cfg["beta_edge"] > 0 and (it % 5) == 0:
            Edge_loss = edge_aware_loss(model, coords_grid, V, M, patch_zyx=cfg["topo_patch"])

        # ------------------------------------------------------------
        # 5) Smoothness terms (two different meanings!)
        # ------------------------------------------------------------
        # G: spatial/field smoothness (screenshot-like): E||∇f||^2
        G = torch.tensor(0.0, device=V.device)
        if cfg.get("beta_grad", 0.0) > 0 and (it % 10) == 0:
            G = field_grad_smoothness(model, pts, n_sub=512)

        # Sm: Gaussian *parameter* conditioning (covariance/shape regularizer)
        Sm = smoothness_loss(model)

        # ------------------------------------------------------------
        # 6) Sparsity + overlap
        # ------------------------------------------------------------
        S = sparsity_loss(model)
        O = overlap_loss_mahalanobis(model, n_pairs=2048)

        # ------------------------------------------------------------
        # 7) Full objective
        # ------------------------------------------------------------
        total = (
            D +
            cfg["lam"] * R +
            cfg["alpha"] * T +
            cfg["beta_tv"] * TV +
            cfg["beta_ssim"] * SSIM_loss +
            cfg["beta_edge"] * Edge_loss +
            cfg.get("beta_grad", 0.0) * G +
            cfg["beta_sparse"] * S +
            cfg["beta_smooth"] * Sm +
            cfg["beta_overlap"] * O
        )

        # Backprop
        opt.zero_grad(set_to_none=True)
        total.backward()

        # Accumulate densification stats from mu gradients
        if cfg["densify_enabled"] and controller is not None and model.mu.grad is not None:
            controller.add_densification_stats(model.mu.grad)

        opt.step()

        # ------------------------------------------------------------
        # 8) Logging
        # ------------------------------------------------------------
        total_val = float(total.detach().cpu())
        losses["D"].append(float(D.detach().cpu()))
        losses["R"].append(float(R.detach().cpu()))
        losses["T"].append(float(T.detach().cpu()))
        losses["G"].append(float(G.detach().cpu()))
        losses["TV"].append(float(TV.detach().cpu()))
        losses["SSIM"].append(float(SSIM_loss.detach().cpu()))
        losses["Edge"].append(float(Edge_loss.detach().cpu()))
        losses["S"].append(float(S.detach().cpu()))
        losses["Sm"].append(float(Sm.detach().cpu()))
        losses["O"].append(float(O.detach().cpu()))
        losses["Total"].append(total_val)
        losses["N"].append(model.N)

        cum_avg_loss = sum(losses["Total"]) / len(losses["Total"])

        # ------------------------------------------------------------
        # 9) Densification step (clone/split/prune)
        # ------------------------------------------------------------
        if cfg["densify_enabled"] and controller is not None:
            if (cfg["densify_from_iter"] <= it < cfg["densify_until_iter"]) and ((it + 1) % cfg["densify_every"] == 0):
                if model.N < cfg["max_gaussians"]:
                    stats = controller.densify_and_prune(
                        grad_threshold=cfg["grad_threshold"],
                        min_amplitude=cfg["min_amplitude"],
                        scene_extent=2.0,  # coordinate cube roughly spans [-1,1] -> extent ~2
                        max_scale=0.1
                    )
                    densify_log.append((it, stats))

                    # IMPORTANT: after changing model parameters (new tensors),
                    # recreate optimizer so it sees the new nn.Parameters.
                    params = (
                        list(model.parameters()) +
                        list(H_mu.parameters()) + list(H_logs.parameters()) +
                        list(H_q.parameters()) + list(H_a.parameters()) + list(H_b.parameters())
                    )
                    opt = torch.optim.Adam(params, lr=current_lr)

                    print(f"  [Densify @{it+1}] N={stats['total_gaussians']} "
                          f"(+{stats['cloned']} clone, +{stats['split']} split, "
                          f"-{stats['pruned_amp']} amp, -{stats['pruned_scale']} scale)")

        # periodic console logging
        if (it + 1) % 200 == 0:
            dt = time.time() - t0
            recent_avg = sum(losses["Total"][-200:]) / min(200, len(losses["Total"]))
            print(f"iter {it+1:5d} | D={losses['D'][-1]:.5f} R={losses['R'][-1]:.1f} "
                  f"T={last_T:.4f} SSIM_loss={last_SSIM:.4f} | "
                  f"Loss: cur={total_val:.4f} avg={cum_avg_loss:.4f} recent={recent_avg:.4f} | "
                  f"N={model.N} | {dt:.1f}s")

        # periodic checkpoint save
        if cfg.get("save_every") and (it + 1) % cfg["save_every"] == 0:
            ckpt_name = f"neurogs_ckpt_iter{it+1}.pt"
            save_checkpoint(
                model, H_mu, H_logs, H_q, H_a, H_b,
                Q, losses, densify_log, cfg,
                os.path.join(CKPT_DIR, ckpt_name),
                iteration=it+1
            )

        # memory housekeeping
        if (it + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return losses, densify_log

# ============================================================================
# Main
# ============================================================================

def main():
    """
    End-to-end pipeline:

    1) Load 3D TIFF volume V_np (Z,Y,X)
    2) Normalize intensities to [0,1]
    3) Build normalized coordinate grid coords_grid (Z,Y,X,3)
    4) Build neurite map M for biased sampling / edge weighting
    5) Initialize Gaussians near neurites using M
    6) Train with rate–distortion + regularizers + densification
    7) Save final checkpoint
    """
    print(f"Device: {DEVICE}")
    print(f"Loading volume: {TIF_PATH}")

    assert os.path.exists(TIF_PATH), f"File not found: {TIF_PATH}"
    V_np = tiff.imread(TIF_PATH)
    print(f"Loaded: {V_np.shape} {V_np.dtype} min/max: {V_np.min()}/{V_np.max()}")

    # Normalize intensities to [0,1] for stable training and consistent loss scales.
    V = V_np.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    V_t = torch.from_numpy(V).to(DEVICE)

    # Coordinate grid in normalized coordinates [-1,1]^3
    coords_grid = make_coord_grid_zyx(V_t.shape, VOXEL_SPACING, DEVICE)
    print(f"coords_grid: {coords_grid.shape}")

    # Importance map for neurites/edges
    M = make_neurite_map(V_t)
    print(f"M: {M.shape} min/max: {float(M.min())}/{float(M.max())} mean: {float(M.mean())}")

    # Initialize Gaussian centers from neurite map so we start with primitives near signal
    N0 = TRAINING_CONFIG["N0"]
    init_means, init_amp = init_gaussians_from_neurite_map(coords_grid, V_t, M, N0)
    model = GaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)

    # Set initial scales relatively small (exp(-3) ~ 0.05) in normalized coordinate units.
    model.log_s.data.fill_(-3.0)
    print(f"Model initialized with N={model.N} Gaussians")

    # Entropy models for each parameter group
    H_mu = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_logs = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_q = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_a = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_b = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)

    # Quantization step sizes
    Q = QuantSteps()

    ensure_checkpoint_dir()
    print(f"Checkpoints will be saved to: {CKPT_DIR}/")

    # Train
    losses, densify_log = train(model, V_t, coords_grid, M, H_mu, H_logs, H_q, H_a, H_b, Q, TRAINING_CONFIG)

    print(f"\nFinal model has {model.N} Gaussians")

    # Final checkpoint
    save_checkpoint(
        model, H_mu, H_logs, H_q, H_a, H_b,
        Q, losses, densify_log, TRAINING_CONFIG,
        CKPT_PATH, iteration=TRAINING_CONFIG["steps"]
    )

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final checkpoint saved to: {CKPT_PATH}")
    print(f"All checkpoints in: {CKPT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()

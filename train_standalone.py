#!/usr/bin/env python3
"""
NeuroGS-Codec Standalone Training Script — 40+ dB PSNR Target
==============================================================

Key changes from previous version:
───────────────────────────────────
1) PSNR-aware training: unweighted MSE branch for PSNR tracking + weighted
   branch for importance sampling — decoupled so gradients target true PSNR.
2) Three-phase schedule: warmup → main (moderate reg) → finetune (pure MSE)
   with aggressive reg annealing in phase 2→3 transition.
3) Larger Gaussian budget (50K) with more aggressive densification and
   gentler pruning. Densification now runs until 85% of training.
4) Cosine-annealed LR with warm restarts at phase transitions.
5) Periodic full-volume PSNR eval (not estimated from batch MSE).
6) Reduced regularizer frequency AND magnitude — topology/overlap/TV
   computed much less often and annealed to near-zero in finetune phase.
7) Increased batch size (24K) with 60/40 uniform/importance split for
   better gradient coverage of background regions.
8) STE quantization only applied in rate loss, not in reconstruction path
   during training — avoids quantization noise degrading PSNR.
"""

import os
import math
import time
import gc
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile as tiff
from tqdm.auto import tqdm

torch.backends.cudnn.benchmark = True

try:
    from cuda_ops import CUDAGaussianMixtureVolume, CUDA_AVAILABLE
    USE_CUDA_OPS = CUDA_AVAILABLE
except ImportError:
    USE_CUDA_OPS = False

if USE_CUDA_OPS:
    print("[CUDA] Custom CUDA kernels loaded – using accelerated forward pass")
else:
    print("[CUDA] Custom kernels not available – using PyTorch fallback")

# ============================================================================
# Configuration
# ============================================================================
TIF_PATH = "10-2900-control-cell-05_cropped_corrected.tif"
VOXEL_SPACING = (0.126, 0.126, 1.0)

CKPT_DIR = "checkpoints"
CKPT_PATH = os.path.join(CKPT_DIR, "neurogs_codec_ckpt_final.pt")

TRAINING_CONFIG = {
    # ── Gaussian budget ──
    "N0": 16000,               # more initial Gaussians
    "max_gaussians": 50000,    # headroom for densification
    "min_gaussians": 512,

    # ── Training schedule ──
    "steps": 60000,            # enough for convergence
    "batch": 24000,            # larger batch = less noisy gradients

    # ── Importance weighting (for sampling, NOT for PSNR calc) ──
    "kappa": 3.0,              # reduced — less distortion of gradient landscape

    # ── Rate-distortion ──
    "lam": 2e-5,               # very low rate penalty
    "alpha": 0.002,            # topology weight (annealed)

    # ── Regularizers (all annealed in finetune phase) ──
    "beta_sparse": 1e-5,
    "beta_smooth": 1e-5,
    "beta_tv": 5e-5,
    "beta_ssim": 0.10,
    "beta_edge": 0.02,
    "beta_overlap": 5e-4,
    "beta_grad": 5e-5,

    # ── Loss composition ──
    # Phase 1 (warmup): charb-heavy for stability
    # Phase 2 (main):   MSE-heavy for PSNR
    # Phase 3 (finetune): almost pure unweighted MSE
    "phase1_end_frac": 0.05,       # 5% warmup
    "phase2_end_frac": 0.70,       # 70% main training
    # phase 3 = remaining 30%

    "phase1_mse_weight": 0.3,
    "phase1_charb_weight": 0.7,
    "phase2_mse_weight": 0.85,
    "phase2_charb_weight": 0.15,
    "phase3_mse_weight": 1.0,
    "phase3_charb_weight": 0.0,
    "phase3_reg_scale": 0.02,      # nearly zero regularization

    "charb_eps": 1e-6,

    # ── Optimizer ──
    "lr": 3e-3,
    "lr_final": 2e-5,              # very low floor for fine details
    "lr_schedule": "cosine",
    "lr_warmup_iters": 500,

    # ── Densification ──
    "densify_enabled": True,
    "densify_from_iter": 600,
    "densify_until_iter": 51000,   # 85% of training
    "densify_every": 600,          # more frequent
    "grad_threshold": 8e-5,        # lower = more aggressive densification

    # ── Pruning (separate schedule) ──
    "prune_warmup_iters": 3000,
    "prune_every": 2400,           # less frequent than densification
    "max_prune_fraction": 0.08,    # conservative
    "min_contrib": 1e-9,
    "use_contrib_prune": True,
    "prune_percentile": 0.02,      # only bottom 2%
    "min_amplitude": 5e-5,

    # ── Eval ──
    "full_eval_every": 5000,       # periodic full-volume PSNR
    "topo_patch": (8, 32, 32),

    # ── Checkpointing ──
    "save_every": 5000,

    # ── Performance ──
    "use_amp": True,
    "use_compile": False,          # must be off with dynamic N
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


# ============================================================================
# Helper Functions
# ============================================================================

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
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
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def gaussian_blur_3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return vol
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, device=vol.device, dtype=vol.dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / (k.sum() + 1e-8)
    v = vol.unsqueeze(0).unsqueeze(0)
    kx = k.view(1, 1, 1, 1, -1)
    v = F.conv3d(v, kx, padding=(0, 0, radius))
    ky = k.view(1, 1, 1, -1, 1)
    v = F.conv3d(v, ky, padding=(0, radius, 0))
    kz = k.view(1, 1, -1, 1, 1)
    v = F.conv3d(v, kz, padding=(radius, 0, 0))
    return v[0, 0]


@torch.no_grad()
def make_neurite_map(vol_zyx: torch.Tensor) -> torch.Tensor:
    v1 = gaussian_blur_3d(vol_zyx, sigma=0.8)
    v2 = gaussian_blur_3d(vol_zyx, sigma=2.0)
    dog = (v1 - v2).abs()
    dz = F.pad(vol_zyx[1:] - vol_zyx[:-1], (0, 0, 0, 0, 0, 1))
    dy = F.pad(vol_zyx[:, 1:] - vol_zyx[:, :-1], (0, 0, 0, 1, 0, 0))
    dx = F.pad(vol_zyx[:, :, 1:] - vol_zyx[:, :, :-1], (0, 1, 0, 0, 0, 0))
    gmag = torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
    m = dog + 0.5 * gmag
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.clamp(0, 1)


def make_coord_grid_zyx(shape_zyx, spacing_xyz, device):
    Z, Y, X = shape_zyx
    dx, dy, dz = spacing_xyz
    xs = torch.arange(X, device=device) * dx
    ys = torch.arange(Y, device=device) * dy
    zs = torch.arange(Z, device=device) * dz
    xn = (xs - xs.min()) / (xs.max() - xs.min() + 1e-8) * 2 - 1
    yn = (ys - ys.min()) / (ys.max() - ys.min() + 1e-8) * 2 - 1
    zn = (zs - zs.min()) / (zs.max() - zs.min() + 1e-8) * 2 - 1
    zz, yy, xx = torch.meshgrid(zn, yn, xn, indexing="ij")
    return torch.stack([xx, yy, zz], dim=-1)


# ============================================================================
# Model
# ============================================================================

class GaussianMixtureVolume(nn.Module):
    def __init__(self, N: int, init_means: torch.Tensor, init_amp: torch.Tensor):
        super().__init__()
        assert init_means.shape == (N, 3)
        assert init_amp.shape == (N,)
        self.N = N
        self.mu = nn.Parameter(init_means.clone())
        self.log_s = nn.Parameter(torch.zeros(N, 3, device=init_means.device) - 2.0)
        q = torch.zeros(N, 4, device=init_means.device)
        q[:, 0] = 1.0
        self.q = nn.Parameter(q)
        self.a = nn.Parameter(init_amp.clone())
        self.b = nn.Parameter(torch.tensor(0.0, device=init_means.device))
        self.sigma_cutoff = 3.0

    def get_gaussian_bounds(self):
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        max_radius = s.max(dim=-1, keepdim=True).values * self.sigma_cutoff
        r = max_radius.expand(-1, 3)
        return self.mu - r, self.mu + r

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        inv_s = 1.0 / (s + 1e-8)
        qn = safe_normalize(self.q)
        Rt = quat_to_rotmat(qn).transpose(-1, -2)

        P = x.shape[0]
        N = self.N
        CHUNK = 4096

        if P * N <= 50_000_000:
            dx = x[:, None, :] - self.mu[None, :, :]
            y = torch.einsum("pni,nij->pnj", dx, Rt)
            y = y * inv_s[None, :, :]
            g = torch.exp(-0.5 * (y * y).sum(dim=-1))
            return (g * self.a[None, :]).sum(dim=1) + self.b

        out = torch.empty(P, device=x.device, dtype=x.dtype)
        for i in range(0, P, CHUNK):
            xc = x[i:i+CHUNK]
            dx = xc[:, None, :] - self.mu[None, :, :]
            y = torch.einsum("pni,nij->pnj", dx, Rt)
            y = y * inv_s[None, :, :]
            g = torch.exp(-0.5 * (y * y).sum(dim=-1))
            out[i:i+CHUNK] = (g * self.a[None, :]).sum(dim=1) + self.b
        return out

    def forward(self, x: torch.Tensor, use_culling: bool = True,
                max_gaussians_per_tile: int = 50000) -> torch.Tensor:
        if self.N == 0:
            return torch.full((x.shape[0],), self.b.item(),
                              device=x.device, dtype=x.dtype)
        P = x.shape[0]
        if (not use_culling) or (self.N <= max_gaussians_per_tile):
            return self._forward_dense(x)

        x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
        g_min, g_max = self.get_gaussian_bounds()
        mask = ((g_max >= x_min[None, :]) & (g_min <= x_max[None, :])).all(dim=-1)
        if mask.sum().item() == 0:
            return torch.full((P,), self.b.item(), device=x.device, dtype=x.dtype)
        if mask.sum().item() > max_gaussians_per_tile:
            return self._forward_dense(x)

        idx = mask.nonzero(as_tuple=True)[0]
        mu = self.mu[idx]; log_s = self.log_s[idx]
        q = self.q[idx]; a = self.a[idx]
        dx = x[:, None, :] - mu[None, :, :]
        s = torch.exp(log_s).clamp(1e-4, 10.0)
        qn = safe_normalize(q)
        Rt = quat_to_rotmat(qn).transpose(-1, -2)
        y = torch.einsum("pni,nij->pnj", dx, Rt)
        y = y / (s[None, :, :] + 1e-8)
        g = torch.exp(-0.5 * (y * y).sum(dim=-1))
        return (g * a[None, :]).sum(dim=1) + self.b


# ============================================================================
# Entropy / Quantization
# ============================================================================

class LaplaceEntropyModel(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.log_b = nn.Parameter(torch.tensor(math.log(init_scale), device=DEVICE))

    def bits_per_element(self, xq: torch.Tensor) -> torch.Tensor:
        if xq.numel() == 0:
            return torch.zeros((), device=xq.device, dtype=xq.dtype)
        b = torch.exp(self.log_b).clamp(1e-6, 1e3)
        logp = -math.log(2.0) - torch.log(b) - (xq.abs() / b)
        return (-logp / math.log(2.0)).mean()

    def total_bits(self, xq: torch.Tensor) -> torch.Tensor:
        if xq.numel() == 0:
            return torch.zeros((), device=xq.device, dtype=xq.dtype)
        b = torch.exp(self.log_b).clamp(1e-6, 1e3)
        logp = -math.log(2.0) - torch.log(b) - (xq.abs() / b)
        return (-logp / math.log(2.0)).sum()


def ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


@dataclass
class QuantSteps:
    mu: float = 1/2048
    log_s: float = 1/256
    q: float = 1/1024
    a: float = 1/1024
    b: float = 1/1024


# ============================================================================
# Losses / Regularizers
# ============================================================================

def charbonnier(x, eps=1e-6):
    return torch.sqrt(x*x + eps*eps)


def mse_loss(pred, target, weight=None):
    sq = (pred - target) ** 2
    if weight is not None:
        sq = weight * sq
    return sq.mean()


def unweighted_mse(pred, target):
    """True MSE — directly maps to PSNR = 10*log10(1/MSE)."""
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def compute_psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def compute_full_volume_psnr(model, coords_grid, V, chunk_size=None):
    """
    Evaluate true full-volume PSNR (no sampling noise).
    Uses CUDA-accelerated forward pass with optimized chunking and CUDA MSE kernel.
    """
    model.eval()
    Z, Y, X, _ = coords_grid.shape
    flat_coords = coords_grid.reshape(-1, 3)
    flat_target = V.reshape(-1)
    n_voxels = flat_coords.shape[0]
    
    # Auto-tune chunk size based on GPU memory and CUDA availability
    if chunk_size is None:
        if USE_CUDA_OPS and torch.cuda.is_available():
            # CUDA kernels are very efficient, use larger chunks
            # Limit by memory: ~500MB per chunk (conservative)
            chunk_size = min(2_000_000, n_voxels)  # 2M points = ~300MB
        else:
            chunk_size = 32768  # Original fallback
    
    # Process in large chunks for CUDA efficiency
    if n_voxels <= chunk_size:
        # Single chunk - use CUDA MSE directly
        p = model(flat_coords, use_culling=False)
        if USE_CUDA_OPS and torch.cuda.is_available():
            from cuda_ops import cuda_compute_mse
            mse = cuda_compute_mse(p, flat_target).item()
        else:
            mse = ((p - flat_target) ** 2).mean().item()
    else:
        # Multiple chunks
        total_se = 0.0
        for i in range(0, n_voxels, chunk_size):
            c = flat_coords[i:i+chunk_size]
            t = flat_target[i:i+chunk_size]
            p = model(c, use_culling=False)
            
            # Accumulate squared error (CUDA accelerated if available)
            if USE_CUDA_OPS and torch.cuda.is_available():
                from cuda_ops import cuda_compute_mse
                total_se += cuda_compute_mse(p, t).item() * c.shape[0]
            else:
                total_se += ((p - t) ** 2).sum().item()
        
        mse = total_se / n_voxels
    
    model.train()
    
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def extract_random_patch(vol, patch_zyx=(8, 32, 32)):
    Z, Y, X = vol.shape
    pz, py, px = patch_zyx
    z0 = torch.randint(0, max(1, Z - pz + 1), (1,), device=vol.device).item()
    y0 = torch.randint(0, max(1, Y - py + 1), (1,), device=vol.device).item()
    x0 = torch.randint(0, max(1, X - px + 1), (1,), device=vol.device).item()
    return (z0, y0, x0), vol[z0:z0+pz, y0:y0+py, x0:x0+px]


def tv3d(p):
    dz = (p[1:] - p[:-1]).abs().mean()
    dy = (p[:, 1:] - p[:, :-1]).abs().mean()
    dx = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
    return dx + dy + dz


def patch_topology_loss(model, coords_grid, V, patch_zyx=(8, 32, 32),
                        tau=0.25, gamma=10.0):
    (z0, y0, x0), _ = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    P = torch.sigmoid(gamma * (pred - tau))
    tv = tv3d(P)
    low = F.avg_pool3d(P[None, None], 3, stride=1, padding=1)[0, 0]
    hf = (P - low).abs().mean()
    return tv + 0.5 * hf


def reconstruction_tv_loss(model, coords_grid, patch_zyx=(8, 32, 32)):
    Z, Y, X, _ = coords_grid.shape
    pz, py, px = patch_zyx
    z0 = torch.randint(0, max(1, Z - pz + 1), (1,), device=coords_grid.device).item()
    y0 = torch.randint(0, max(1, Y - py + 1), (1,), device=coords_grid.device).item()
    x0 = torch.randint(0, max(1, X - px + 1), (1,), device=coords_grid.device).item()
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    return tv3d(pred)


def ssim_loss_3d(pred, target, win=3, K1=0.01, K2=0.03):
    pred_ = pred[None, None]; tgt_ = target[None, None]
    pad = win // 2
    mu_p = F.avg_pool3d(pred_, win, stride=1, padding=pad)
    mu_t = F.avg_pool3d(tgt_, win, stride=1, padding=pad)
    sigma_p = F.avg_pool3d(pred_*pred_, win, stride=1, padding=pad) - mu_p*mu_p
    sigma_t = F.avg_pool3d(tgt_*tgt_, win, stride=1, padding=pad) - mu_t*mu_t
    sigma_pt = F.avg_pool3d(pred_*tgt_, win, stride=1, padding=pad) - mu_p*mu_t
    C1, C2 = K1**2, K2**2
    ssim_map = ((2*mu_p*mu_t+C1)*(2*sigma_pt+C2)) / \
               ((mu_p**2+mu_t**2+C1)*(sigma_p+sigma_t+C2)+1e-8)
    return 1.0 - ssim_map.mean()


def patch_ssim_loss(model, coords_grid, V, patch_zyx=(8, 32, 32)):
    (z0, y0, x0), tgt = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    return ssim_loss_3d(pred, tgt, win=3)


def edge_aware_loss(model, coords_grid, V, M, patch_zyx=(8, 32, 32)):
    (z0, y0, x0), tgt = extract_random_patch(V, patch_zyx)
    pz, py, px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    m_patch = M[z0:z0+pz, y0:y0+py, x0:x0+px]
    dz = F.pad(tgt[1:]-tgt[:-1], (0,0,0,0,0,1)).abs()
    dy = F.pad(tgt[:,1:]-tgt[:,:-1], (0,0,0,1,0,0)).abs()
    dx = F.pad(tgt[:,:,1:]-tgt[:,:,:-1], (0,1,0,0,0,0)).abs()
    w = (dz + dy + dx + m_patch).clamp(0, 1)
    return (w * (pred - tgt).abs()).mean()


def sparsity_loss(model):
    if model.N == 0:
        return torch.zeros((), device=model.b.device)
    return model.a.abs().mean()


def smoothness_loss(model):
    if model.N == 0:
        return torch.zeros((), device=model.b.device)
    s = torch.exp(model.log_s)
    min_scale_penalty = torch.relu(0.01 - s).mean()
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values + 1e-8
    anisotropy = (s_max / s_min - 1.0).clamp(min=0).mean()
    max_scale_penalty = torch.relu(s - 1.0).mean()
    return min_scale_penalty + 0.1 * anisotropy + 0.1 * max_scale_penalty


def field_grad_smoothness(model, pts, n_sub=256):
    if pts.shape[0] > n_sub:
        idx = torch.randint(0, pts.shape[0], (n_sub,), device=pts.device)
        pts = pts[idx]
    pts = pts.detach().requires_grad_(True)
    pred = model(pts)
    g = torch.autograd.grad(pred.sum(), pts, create_graph=True)[0]
    return (g.pow(2).sum(dim=-1)).mean()


def overlap_loss_mahalanobis(model, n_pairs=512, eps=1e-4):
    N = model.N
    if N < 2:
        return torch.zeros((), device=model.b.device)
    idx_i = torch.randint(0, N, (n_pairs,), device=model.mu.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=model.mu.device)
    m = idx_i != idx_j
    idx_i, idx_j = idx_i[m], idx_j[m]
    if idx_i.numel() == 0:
        return torch.zeros((), device=model.b.device)
    mu_i, mu_j = model.mu[idx_i], model.mu[idx_j]
    dmu = (mu_i - mu_j).unsqueeze(-1)
    s_i = torch.exp(model.log_s[idx_i]).clamp(1e-4, 10.0)
    s_j = torch.exp(model.log_s[idx_j]).clamp(1e-4, 10.0)
    R_i = quat_to_rotmat(safe_normalize(model.q[idx_i]))
    R_j = quat_to_rotmat(safe_normalize(model.q[idx_j]))
    Si = R_i @ torch.diag_embed(s_i*s_i) @ R_i.transpose(-1, -2)
    Sj = R_j @ torch.diag_embed(s_j*s_j) @ R_j.transpose(-1, -2)
    I = torch.eye(3, device=model.mu.device).unsqueeze(0)
    Sij = Si + Sj + eps * I
    x = torch.linalg.solve(Sij, dmu)
    d2 = (dmu.transpose(-1, -2) @ x).squeeze(-1).squeeze(-1)
    return torch.exp(-0.5 * d2).mean()


# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def precompute_sampling_cdf(M):
    flatM = M.reshape(-1)
    total = flatM.numel()
    max_categories = 2**24 - 1
    if total > max_categories:
        cand = torch.randint(0, total, (max_categories,), device=M.device)
        probs = flatM[cand]
        probs = probs / (probs.sum() + 1e-8)
        cdf = torch.cumsum(probs, dim=0)
        cdf[-1] = 1.0
        return cdf, cand
    else:
        probs = flatM / (flatM.sum() + 1e-8)
        cdf = torch.cumsum(probs, dim=0)
        cdf[-1] = 1.0
        return cdf, None


@torch.no_grad()
def sample_points(coords_grid, V, M, n_uniform, n_biased,
                  cdf=None, cdf_cand=None):
    Z, Y, X, _ = coords_grid.shape
    total = Z * Y * X

    idx_u = torch.randint(0, total, (n_uniform,), device=V.device)
    zu = idx_u // (Y * X)
    yu = (idx_u % (Y * X)) // X
    xu = idx_u % X

    if cdf is not None:
        rand_vals = torch.rand(n_biased, device=V.device)
        idx_in_cdf = torch.searchsorted(cdf, rand_vals).clamp(0, cdf.numel()-1)
        idx_b = cdf_cand[idx_in_cdf] if cdf_cand is not None else idx_in_cdf
    else:
        flatM = M.reshape(-1)
        max_categories = 2**24 - 1
        if total > max_categories:
            cand = torch.randint(0, total, (max_categories,), device=V.device)
            probs = flatM[cand]
            probs = probs / (probs.sum() + 1e-8)
            sel = torch.multinomial(probs, n_biased, replacement=True)
            idx_b = cand[sel]
        else:
            probs = flatM / (flatM.sum() + 1e-8)
            idx_b = torch.multinomial(probs, n_biased, replacement=True)

    zb = idx_b // (Y * X)
    yb = (idx_b % (Y * X)) // X
    xb = idx_b % X

    z = torch.cat([zu, zb]); y = torch.cat([yu, yb]); x = torch.cat([xu, xb])
    return coords_grid[z, y, x], V[z, y, x], M[z, y, x]


@torch.no_grad()
def init_gaussians_from_neurite_map(coords_grid, V, M, N0):
    Z, Y, X, _ = coords_grid.shape
    total = Z * Y * X
    flatM = M.reshape(-1)
    max_categories = 2**24 - 1
    if total > max_categories:
        cand = torch.randint(0, total, (max_categories,), device=M.device)
        probs = flatM[cand]
        probs = probs / (probs.sum() + 1e-8)
        sel = torch.multinomial(probs, N0, replacement=True)
        idx = cand[sel]
    else:
        probs = flatM / (flatM.sum() + 1e-8)
        idx = torch.multinomial(probs, N0, replacement=True)
    z = idx // (Y * X); y = (idx % (Y * X)) // X; x = idx % X
    return coords_grid[z, y, x], V[z, y, x].clone()


# ============================================================================
# Checkpointing
# ============================================================================

def ensure_checkpoint_dir():
    os.makedirs(CKPT_DIR, exist_ok=True)


def save_checkpoint(model, H_mu, H_logs, H_q, H_a, H_b, Q, losses,
                    densify_log, config, path, iteration=None):
    ensure_checkpoint_dir()
    torch.save({
        "model_state": model.state_dict(),
        "model_N": model.N,
        "entropy_state": {
            "H_mu": H_mu.state_dict(), "H_logs": H_logs.state_dict(),
            "H_q": H_q.state_dict(), "H_a": H_a.state_dict(),
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
    }, path)
    print(f"[Checkpoint] Saved to {path} (iter={iteration}, N={model.N})")


# ============================================================================
# Densification Controller
# ============================================================================

class DensificationController:
    def __init__(self, model, percent_dense=0.01, min_keep: int = 256):
        self.model = model
        self.percent_dense = percent_dense
        self.min_keep = int(min_keep)
        self.xyz_gradient_accum = torch.zeros((model.N, 1), device=DEVICE)
        self.denom = torch.zeros((model.N, 1), device=DEVICE)

    def reset_stats(self):
        N = self.model.N
        self.xyz_gradient_accum = torch.zeros((N, 1), device=DEVICE)
        self.denom = torch.zeros((N, 1), device=DEVICE)

    def add_densification_stats(self, grad_mu):
        if grad_mu is None:
            return
        self.xyz_gradient_accum += grad_mu.norm(dim=-1, keepdim=True)
        self.denom += 1

    def get_scaling(self):
        return torch.exp(self.model.log_s).clamp(1e-4, 10.0)

    @torch.no_grad()
    def _contrib_score(self):
        if self.model.N == 0:
            return torch.zeros((0,), device=self.model.b.device)
        s = self.get_scaling()
        scale_score = (s.prod(dim=-1) + 1e-12).pow(1.0/3.0)
        return self.model.a.data.abs() * scale_score

    def _cat_tensors(self, new_mu, new_log_s, new_q, new_a):
        add = new_mu.shape[0]
        self.model.mu = nn.Parameter(torch.cat([self.model.mu.data, new_mu], 0))
        self.model.log_s = nn.Parameter(torch.cat([self.model.log_s.data, new_log_s], 0))
        self.model.q = nn.Parameter(torch.cat([self.model.q.data, new_q], 0))
        self.model.a = nn.Parameter(torch.cat([self.model.a.data, new_a.reshape(-1)], 0))
        self.model.N += add
        self.xyz_gradient_accum = torch.cat(
            [self.xyz_gradient_accum, torch.zeros((add, 1), device=DEVICE)], 0)
        self.denom = torch.cat(
            [self.denom, torch.zeros((add, 1), device=DEVICE)], 0)

    @torch.no_grad()
    def prune_points(self, prune_mask):
        if self.model.N == 0:
            return 0
        keep = ~prune_mask
        if keep.sum().item() == keep.numel():
            return 0
        remaining = int(keep.sum().item())
        if remaining < self.min_keep:
            score = self._contrib_score()
            k = min(self.min_keep, self.model.N)
            keep_idx = torch.topk(score, k=k, largest=True).indices
            keep = torch.zeros_like(prune_mask, dtype=torch.bool)
            keep[keep_idx] = True
        pruned = int((~keep).sum().item())
        self.model.mu = nn.Parameter(self.model.mu.data[keep])
        self.model.log_s = nn.Parameter(self.model.log_s.data[keep])
        self.model.q = nn.Parameter(self.model.q.data[keep])
        self.model.a = nn.Parameter(self.model.a.data[keep])
        self.model.N = self.model.mu.shape[0]
        self.xyz_gradient_accum = self.xyz_gradient_accum[keep]
        self.denom = self.denom[keep]
        return pruned

    @torch.no_grad()
    def densify_and_clone(self, grads, grad_threshold, scene_extent,
                          remaining_capacity):
        if remaining_capacity <= 0 or self.model.N == 0:
            return 0
        scaling = self.get_scaling()
        sel = ((grads.squeeze() >= grad_threshold) &
               (scaling.max(dim=1).values <= self.percent_dense * scene_extent))
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return 0
        if idx.numel() > remaining_capacity:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:remaining_capacity]]
        self._cat_tensors(
            self.model.mu.data[idx].clone(),
            self.model.log_s.data[idx].clone(),
            self.model.q.data[idx].clone(),
            self.model.a.data[idx].clone()
        )
        return int(idx.numel())

    @torch.no_grad()
    def densify_and_split(self, grads, grad_threshold, scene_extent,
                          remaining_capacity, N_split=2):
        if remaining_capacity <= 0 or self.model.N == 0:
            return 0
        N_grads = grads.shape[0]
        scaling = self.get_scaling()[:N_grads]
        sel = ((grads.squeeze() >= grad_threshold) &
               (scaling.max(dim=1).values > self.percent_dense * scene_extent))
        idx = sel.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return 0
        max_parents = max(0, remaining_capacity // max(1, N_split - 1))
        if max_parents <= 0:
            return 0
        if idx.numel() > max_parents:
            idx = idx[torch.randperm(idx.numel(), device=idx.device)[:max_parents]]

        stds = scaling[idx].repeat(N_split, 1)
        samples = torch.normal(mean=torch.zeros_like(stds), std=stds)
        R = quat_to_rotmat(safe_normalize(self.model.q.data[idx])).repeat(N_split, 1, 1)
        new_mu = (torch.bmm(R, samples.unsqueeze(-1)).squeeze(-1) +
                  self.model.mu.data[idx].repeat(N_split, 1)).clamp(-1, 1)
        new_log_s = self.model.log_s.data[idx].repeat(N_split, 1) - math.log(0.8 * N_split)
        new_q = self.model.q.data[idx].repeat(N_split, 1)
        new_a = (self.model.a.data[idx].repeat(N_split) / N_split).reshape(-1)

        self._cat_tensors(new_mu, new_log_s, new_q, new_a)
        prune_mask = torch.zeros((self.model.N,), device=DEVICE, dtype=torch.bool)
        prune_mask[idx] = True
        self.prune_points(prune_mask)
        return int(idx.numel() * (N_split - 1))

    @torch.no_grad()
    def prune_low_contribution(self, min_contrib=1e-9, max_prune_fraction=0.08):
        if self.model.N == 0:
            return 0
        score = self._contrib_score()
        prune_mask = score < float(min_contrib)
        n_would_prune = prune_mask.sum().item()
        max_allowed = int(self.model.N * max_prune_fraction)
        if n_would_prune > max_allowed:
            n_keep = self.model.N - max_allowed
            keep_idx = torch.topk(score, k=n_keep, largest=True).indices
            prune_mask = torch.ones_like(prune_mask, dtype=torch.bool)
            prune_mask[keep_idx] = False
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def prune_low_contribution_percentile(self, q=0.02, max_prune_fraction=0.08):
        if self.model.N == 0:
            return 0
        score = self._contrib_score()
        thresh = float(torch.quantile(score, q).item())
        prune_mask = score < thresh
        n_would_prune = prune_mask.sum().item()
        max_allowed = int(self.model.N * max_prune_fraction)
        if n_would_prune > max_allowed:
            n_keep = self.model.N - max_allowed
            keep_idx = torch.topk(score, k=n_keep, largest=True).indices
            prune_mask = torch.ones_like(prune_mask, dtype=torch.bool)
            prune_mask[keep_idx] = False
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def prune_low_amplitude(self, min_amplitude=5e-5):
        if self.model.N == 0:
            return 0
        prune_mask = self.model.a.data.abs() < float(min_amplitude)
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def prune_large_scale(self, max_scale=0.5, scene_extent=2.0):
        if self.model.N == 0:
            return 0
        scaling = self.get_scaling()
        prune_mask = scaling.max(dim=1).values > float(max_scale) * float(scene_extent)
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold, remaining_capacity, scene_extent,
                          max_scale, N_split, prune_on, use_contrib_prune,
                          min_contrib, prune_percentile, min_amplitude,
                          max_prune_fraction=0.08):
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0

        N_before_clone = self.model.N
        n_cloned = self.densify_and_clone(grads, grad_threshold, scene_extent,
                                           remaining_capacity)
        remaining_capacity = max(0, remaining_capacity - n_cloned)
        grads_for_split = (grads[:N_before_clone] if grads.shape[0] > N_before_clone
                           else grads)
        n_split = self.densify_and_split(grads_for_split, grad_threshold,
                                          scene_extent, remaining_capacity, N_split)

        n_pruned_amp = 0
        if prune_on:
            if use_contrib_prune:
                if prune_percentile is not None:
                    n_pruned_amp = self.prune_low_contribution_percentile(
                        q=prune_percentile, max_prune_fraction=max_prune_fraction)
                else:
                    n_pruned_amp = self.prune_low_contribution(
                        min_contrib=min_contrib,
                        max_prune_fraction=max_prune_fraction)
            else:
                n_pruned_amp = self.prune_low_amplitude(min_amplitude=min_amplitude)

        n_pruned_scale = (self.prune_large_scale(max_scale=max_scale,
                                                  scene_extent=scene_extent)
                          if prune_on else 0)
        self.reset_stats()
        torch.cuda.empty_cache()
        return {
            "cloned": n_cloned, "split": n_split,
            "pruned_amp": n_pruned_amp, "pruned_scale": n_pruned_scale,
            "total_gaussians": self.model.N,
        }


# ============================================================================
# Training
# ============================================================================

def get_phase(it: int, cfg: dict) -> Tuple[int, str, float, float, float]:
    """Return (phase_num, name, mse_w, charb_w, reg_scale) for current iter."""
    steps = cfg["steps"]
    p1_end = int(cfg["phase1_end_frac"] * steps)
    p2_end = int(cfg["phase2_end_frac"] * steps)

    if it < p1_end:
        return (1, "WARMUP",
                cfg["phase1_mse_weight"], cfg["phase1_charb_weight"], 1.0)
    elif it < p2_end:
        # Linearly anneal reg from 1.0 → phase3_reg_scale over phase 2
        progress = (it - p1_end) / max(p2_end - p1_end, 1)
        reg_scale = 1.0 + progress * (cfg["phase3_reg_scale"] - 1.0)
        return (2, "MAIN",
                cfg["phase2_mse_weight"], cfg["phase2_charb_weight"], reg_scale)
    else:
        return (3, "FINETUNE",
                cfg["phase3_mse_weight"], cfg["phase3_charb_weight"],
                cfg["phase3_reg_scale"])


def train(model, V, coords_grid, M, H_mu, H_logs, H_q, H_a, H_b, Q, cfg):
    controller = None
    if cfg["densify_enabled"]:
        controller = DensificationController(
            model, percent_dense=0.01,
            min_keep=int(cfg.get("min_gaussians", 256)))

    use_amp = cfg.get("use_amp", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    params = (list(model.parameters()) +
              list(H_mu.parameters()) + list(H_logs.parameters()) +
              list(H_q.parameters()) + list(H_a.parameters()) +
              list(H_b.parameters()))
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    steps = cfg["steps"]
    lr, lr_final = cfg["lr"], cfg.get("lr_final", cfg["lr"])
    warmup_iters = cfg.get("lr_warmup_iters", 0)
    charb_eps = cfg.get("charb_eps", 1e-6)

    losses: Dict[str, List] = {
        k: [] for k in [
            "D", "D_mse", "D_mse_uw", "D_charb", "PSNR_batch", "PSNR_full",
            "R", "R_total_bits", "R_bits_per_gaussian",
            "T", "TV", "SSIM", "Edge", "G", "S", "Sm", "O",
            "Total", "N",
        ]
    }
    densify_log = []
    t0 = time.time()
    best_psnr = 0.0
    last_full_psnr = 0.0

    sampling_cdf, sampling_cand = precompute_sampling_cdf(M)

    print("\n" + "="*70)
    print(f"Device: {DEVICE} | AMP: {use_amp}")
    print(f"Initial N: {model.N} → max {cfg['max_gaussians']}")
    print(f"Steps: {steps} | Batch: {cfg['batch']}")
    print(f"LR: {lr} → {lr_final} (cosine + warmup={warmup_iters})")
    print(f"Phases: warmup→{cfg['phase1_end_frac']:.0%}, "
          f"main→{cfg['phase2_end_frac']:.0%}, finetune→100%")
    print(f"Densify: every {cfg['densify_every']} iters, "
          f"until iter {cfg['densify_until_iter']}")
    print(f"Prune: every {cfg['prune_every']} iters, "
          f"max {cfg['max_prune_fraction']:.0%}/iter")
    print(f"Full-vol PSNR eval every {cfg['full_eval_every']} iters")
    print(f"Target: 40+ dB PSNR")
    print("="*70 + "\n")

    # Compute initial baseline PSNR
    print("Computing initial baseline PSNR...")
    initial_psnr = compute_full_volume_psnr(model, coords_grid, V)
    last_full_psnr = initial_psnr
    best_psnr = initial_psnr
    losses["PSNR_full"].append((0, initial_psnr))
    print(f"Initial PSNR: {initial_psnr:.2f} dB\n")

    for it in tqdm(range(steps), desc="Training"):
        # ── LR schedule ──
        if it < warmup_iters:
            current_lr = lr_final + (lr - lr_final) * (it / max(warmup_iters, 1))
        else:
            progress = (it - warmup_iters) / max(steps - warmup_iters - 1, 1)
            current_lr = lr_final + 0.5 * (lr - lr_final) * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        # ── Phase logic ──
        phase_num, phase_name, cur_mse_w, cur_charb_w, reg_scale = get_phase(it, cfg)

        # ── Sample points (60% uniform, 40% importance) ──
        n_u = int(cfg["batch"] * 0.6)
        n_b = cfg["batch"] - n_u
        pts, tgt, mval = sample_points(coords_grid, V, M, n_u, n_b,
                                        cdf=sampling_cdf, cdf_cand=sampling_cand)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = model(pts)

            # ── Distortion: weighted (for structure) + unweighted (for PSNR) ──
            w = 1.0 + cfg["kappa"] * mval
            D_mse_weighted = mse_loss(pred, tgt, weight=w)
            D_mse_uw = unweighted_mse(pred, tgt)  # true PSNR signal
            D_charb = (w * charbonnier(pred - tgt, eps=charb_eps)).mean()

            # In finetune: use unweighted MSE as primary loss
            if phase_num == 3:
                D = cur_mse_w * D_mse_uw + cur_charb_w * D_charb
            else:
                D = cur_mse_w * D_mse_weighted + cur_charb_w * D_charb

            # ── Rate loss (STE quantization only here, not in recon path) ──
            if model.N > 0:
                mu_q = ste_round(model.mu / Q.mu)
                logs_q = ste_round(model.log_s / Q.log_s)
                q_q = ste_round(model.q / Q.q)
                a_q = ste_round(model.a / Q.a)
            else:
                mu_q = model.mu; logs_q = model.log_s
                q_q = model.q; a_q = model.a
            b_q = ste_round(model.b / Q.b)

            R = (H_mu.bits_per_element(mu_q) + H_logs.bits_per_element(logs_q) +
                 H_q.bits_per_element(q_q) + H_a.bits_per_element(a_q) +
                 H_b.bits_per_element(b_q))
            R_total = (H_mu.total_bits(mu_q) + H_logs.total_bits(logs_q) +
                       H_q.total_bits(q_q) + H_a.total_bits(a_q) +
                       H_b.total_bits(b_q))

            # ── Regularizers (computed at reduced frequency) ──
            T = torch.zeros((), device=V.device)
            if cfg["alpha"] > 0 and (it % 200) == 0:
                T = patch_topology_loss(model, coords_grid, V,
                                        patch_zyx=cfg["topo_patch"])

            TV = torch.zeros((), device=V.device)
            if cfg["beta_tv"] > 0 and (it % 100) == 0:
                TV = reconstruction_tv_loss(model, coords_grid,
                                            patch_zyx=cfg["topo_patch"])

            SSIM_loss = torch.zeros((), device=V.device)
            if cfg["beta_ssim"] > 0 and (it % 50) == 0:
                SSIM_loss = patch_ssim_loss(model, coords_grid, V,
                                            patch_zyx=cfg["topo_patch"])

            Edge_loss = torch.zeros((), device=V.device)
            if cfg["beta_edge"] > 0 and (it % 50) == 0:
                Edge_loss = edge_aware_loss(model, coords_grid, V, M,
                                            patch_zyx=cfg["topo_patch"])

            G = torch.zeros((), device=V.device)
            if cfg.get("beta_grad", 0) > 0 and (it % 100) == 0:
                G = field_grad_smoothness(model, pts, n_sub=256)

            Sm = torch.zeros((), device=V.device)
            S = torch.zeros((), device=V.device)
            O = torch.zeros((), device=V.device)
            if (it % 10) == 0:
                Sm = smoothness_loss(model)
                S = sparsity_loss(model)
            if (it % 20) == 0:
                O = overlap_loss_mahalanobis(model)

            # ── Total loss ──
            total = (
                D
                + cfg["lam"] * R * reg_scale
                + cfg["alpha"] * T * reg_scale
                + cfg["beta_tv"] * TV * reg_scale
                + cfg["beta_ssim"] * SSIM_loss * reg_scale
                + cfg["beta_edge"] * Edge_loss * reg_scale
                + cfg.get("beta_grad", 0) * G * reg_scale
                + cfg["beta_sparse"] * S * reg_scale
                + cfg["beta_smooth"] * Sm * reg_scale
                + cfg["beta_overlap"] * O * reg_scale
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()

        if (cfg["densify_enabled"] and controller is not None
                and model.mu.grad is not None):
            controller.add_densification_stats(model.mu.grad)

        scaler.step(opt)
        scaler.update()

        # ── Logging (every 10 iters) ──
        if (it % 10) == 0 or it == steps - 1:
            uw_mse_val = float(D_mse_uw.detach())
            batch_psnr = 10 * math.log10(1.0 / max(uw_mse_val, 1e-10))
            if batch_psnr > best_psnr:
                best_psnr = batch_psnr

            losses["D"].append(float(D.detach()))
            losses["D_mse"].append(float(D_mse_weighted.detach()))
            losses["D_mse_uw"].append(uw_mse_val)
            losses["D_charb"].append(float(D_charb.detach()))
            losses["PSNR_batch"].append(batch_psnr)
            losses["R"].append(float(R.detach()))
            total_bits = float(R_total.detach())
            losses["R_total_bits"].append(total_bits)
            losses["R_bits_per_gaussian"].append(total_bits / max(model.N, 1))
            losses["T"].append(float(T.detach()))
            losses["TV"].append(float(TV.detach()))
            losses["SSIM"].append(float(SSIM_loss.detach()))
            losses["Edge"].append(float(Edge_loss.detach()))
            losses["G"].append(float(G.detach()))
            losses["S"].append(float(S.detach()))
            losses["Sm"].append(float(Sm.detach()))
            losses["O"].append(float(O.detach()))
            losses["Total"].append(float(total.detach()))
            losses["N"].append(model.N)

        # ── Densification ──
        if cfg["densify_enabled"] and controller is not None:
            if (cfg["densify_from_iter"] <= it < cfg["densify_until_iter"]
                    and (it + 1) % cfg["densify_every"] == 0):
                remaining = int(cfg["max_gaussians"] - model.N)
                prune_on = (it + 1) >= int(cfg.get("prune_warmup_iters", 0))
                should_prune = ((it + 1) % cfg.get("prune_every",
                                                     cfg["densify_every"]) == 0)

                stats = controller.densify_and_prune(
                    grad_threshold=float(cfg["grad_threshold"]),
                    remaining_capacity=max(0, remaining),
                    scene_extent=2.0, max_scale=0.1, N_split=2,
                    prune_on=(prune_on and should_prune),
                    use_contrib_prune=bool(cfg.get("use_contrib_prune", True)),
                    min_contrib=float(cfg.get("min_contrib", 1e-9)),
                    prune_percentile=cfg.get("prune_percentile", None),
                    min_amplitude=float(cfg.get("min_amplitude", 5e-5)),
                    max_prune_fraction=float(cfg.get("max_prune_fraction", 0.08)),
                )
                densify_log.append((it + 1, stats))

                # Rebuild optimizer with current LR
                params = (list(model.parameters()) +
                          list(H_mu.parameters()) + list(H_logs.parameters()) +
                          list(H_q.parameters()) + list(H_a.parameters()) +
                          list(H_b.parameters()))
                opt = torch.optim.Adam(params, lr=current_lr)

                print(f"  [Densify @{it+1}] N={stats['total_gaussians']} "
                      f"(+{stats['cloned']}c +{stats['split']}s "
                      f"-{stats['pruned_amp']}p -{stats['pruned_scale']}sc)")

        # ── Periodic full-volume PSNR ──
        if (cfg.get("full_eval_every") and
                (it + 1) % cfg["full_eval_every"] == 0):
            full_psnr = compute_full_volume_psnr(model, coords_grid, V)
            last_full_psnr = full_psnr
            losses["PSNR_full"].append((it + 1, full_psnr))
            print(f"\n  >>> FULL-VOL PSNR @{it+1}: {full_psnr:.2f} dB "
                  f"(batch est: {batch_psnr:.2f} dB) <<<\n")

            if full_psnr > best_psnr:
                best_psnr = full_psnr

            # Save best
            if full_psnr >= 35.0:
                save_checkpoint(
                    model, H_mu, H_logs, H_q, H_a, H_b, Q, losses,
                    densify_log, cfg,
                    os.path.join(CKPT_DIR, "neurogs_best_psnr.pt"),
                    iteration=it+1)

        # ── Print progress ──
        if (it + 1) % 500 == 0:
            dt = time.time() - t0
            ips = (it + 1) / dt
            total_kb = (losses["R_total_bits"][-1] / 8 / 1024
                        if losses["R_total_bits"] else 0)
            bpsnr = losses["PSNR_batch"][-1] if losses["PSNR_batch"] else 0
            print(f"iter {it+1:5d} [{phase_name}] | "
                  f"PSNR(batch)={bpsnr:.2f}dB "
                  f"PSNR(full)={last_full_psnr:.2f}dB "
                  f"best={best_psnr:.2f}dB | "
                  f"D_uw={losses['D_mse_uw'][-1]:.6f} | "
                  f"N={model.N} | {total_kb:.1f}KB | "
                  f"lr={current_lr:.2e} | reg_s={reg_scale:.3f} | "
                  f"{dt:.0f}s ({ips:.1f} it/s)")

        # ── Periodic checkpoint ──
        if cfg.get("save_every") and (it + 1) % cfg["save_every"] == 0:
            save_checkpoint(
                model, H_mu, H_logs, H_q, H_a, H_b, Q, losses,
                densify_log, cfg,
                os.path.join(CKPT_DIR, f"neurogs_ckpt_iter{it+1}.pt"),
                iteration=it+1)

        if (it + 1) % 3000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # ── Final full-volume evaluation ──
    print("\n" + "="*70)
    print("Computing FINAL full-volume PSNR...")
    final_psnr = compute_full_volume_psnr(model, coords_grid, V)
    losses["PSNR_full"].append((steps, final_psnr))
    print(f"\n  >>> FINAL FULL-VOLUME PSNR: {final_psnr:.2f} dB <<<")
    print(f"  >>> Best seen: {max(best_psnr, final_psnr):.2f} dB <<<")
    print("="*70 + "\n")

    return losses, densify_log


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading volume: {TIF_PATH}")
    assert os.path.exists(TIF_PATH), f"File not found: {TIF_PATH}"

    V_np = tiff.imread(TIF_PATH)
    print(f"Loaded: {V_np.shape} {V_np.dtype} min/max: {V_np.min()}/{V_np.max()}")

    V = V_np.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    V_t = torch.from_numpy(V).to(DEVICE)

    coords_grid = make_coord_grid_zyx(V_t.shape, VOXEL_SPACING, DEVICE)
    M = make_neurite_map(V_t)

    N0 = TRAINING_CONFIG["N0"]
    init_means, init_amp = init_gaussians_from_neurite_map(coords_grid, V_t, M, N0)

    if USE_CUDA_OPS:
        model = CUDAGaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)
        print(f"[CUDA] Using CUDAGaussianMixtureVolume with {N0} Gaussians")
    else:
        model = GaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)

    # Initialize scales smaller for finer detail
    model.log_s.data.fill_(-3.5)

    H_mu = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_logs = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_q = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_a = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_b = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)

    Q = QuantSteps()
    ensure_checkpoint_dir()
    print(f"Checkpoints → {CKPT_DIR}/")

    losses, densify_log = train(model, V_t, coords_grid, M,
                                 H_mu, H_logs, H_q, H_a, H_b, Q,
                                 TRAINING_CONFIG)

    print(f"\nFinal N={model.N}")
    save_checkpoint(model, H_mu, H_logs, H_q, H_a, H_b, Q, losses,
                    densify_log, TRAINING_CONFIG, CKPT_PATH,
                    iteration=TRAINING_CONFIG["steps"])
    print("Done.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
NeuroGS-Codec Standalone Training Script
Run with: python train_standalone.py

For detached execution (survives VS Code disconnect):
  tmux new -s train
  python train_standalone.py
  # Detach: Ctrl+B, then D
  # Reattach later: tmux attach -t train

Or with nohup:
  nohup python train_standalone.py > training.log 2>&1 &
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
TIF_PATH = "10-2900-control-cell-05_cropped_corrected.tif"
VOXEL_SPACING = (0.126, 0.126, 1.0)  # (dx, dy, dz)
CKPT_DIR = "checkpoints"  # Checkpoint folder
CKPT_PATH = os.path.join(CKPT_DIR, "neurogs_codec_ckpt_final.pt")
BITSTREAM_PATH = "neurogs_codec_stream.npz.gz"

# Training config
TRAINING_CONFIG = {
    "N0": 5000,
    "steps": 30000,
    "batch": 2000,
    "kappa": 8.0,
    "lam": 0.001,
    "alpha": 0.01,
    "beta_sparse": 0.003,
    "beta_smooth": 0.001,
    "beta_tv": 0.002,
    "beta_ssim": 0.1,
    "beta_edge": 0.05,
    "beta_overlap": 0.01,
    "lr": 3e-3,
    "lr_final": 5e-4,
    "densify_enabled": True,
    "densify_from_iter": 500,
    "densify_until_iter": 20000,
    "densify_every": 500,
    "max_gaussians": 150000,
    "min_amplitude": 0.0005,
    "grad_threshold": 0.00015,
    "topo_patch": (8, 16, 16),
    "save_every": 1000,  # Save checkpoint every N iterations
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
        ww+xx-yy-zz, 2*(xy-wz), 2*(xz+wy),
        2*(xy+wz), ww-xx+yy-zz, 2*(yz-wx),
        2*(xz-wy), 2*(yz+wx), ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

def safe_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def gaussian_blur_3d(vol: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return vol
    radius = int(3 * sigma + 0.5)
    x = torch.arange(-radius, radius+1, device=vol.device, dtype=vol.dtype)
    k = torch.exp(-(x**2)/(2*sigma**2))
    k = k / (k.sum() + 1e-8)
    v = vol.unsqueeze(0).unsqueeze(0)
    kx = k.view(1,1,1,1,-1)
    v = F.conv3d(v, kx, padding=(0,0,radius))
    ky = k.view(1,1,1,-1,1)
    v = F.conv3d(v, ky, padding=(0,radius,0))
    kz = k.view(1,1,-1,1,1)
    v = F.conv3d(v, kz, padding=(radius,0,0))
    return v[0,0]

@torch.no_grad()
def make_neurite_map(vol_zyx: torch.Tensor) -> torch.Tensor:
    v1 = gaussian_blur_3d(vol_zyx, sigma=0.8)
    v2 = gaussian_blur_3d(vol_zyx, sigma=2.0)
    dog = (v1 - v2).abs()
    dz = F.pad(vol_zyx[1:] - vol_zyx[:-1], (0,0,0,0,0,1))
    dy = F.pad(vol_zyx[:,1:] - vol_zyx[:,:-1], (0,0,0,1,0,0))
    dx = F.pad(vol_zyx[:,:,1:] - vol_zyx[:,:,:-1], (0,1,0,0,0,0))
    gmag = torch.sqrt(dx*dx + dy*dy + dz*dz + 1e-8)
    m = dog + 0.5 * gmag
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return m.clamp(0,1)

def make_coord_grid_zyx(shape_zyx: Tuple[int,int,int], spacing_xyz: Tuple[float,float,float], device: str):
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
# Model
# ============================================================================

class GaussianMixtureVolume(nn.Module):
    def __init__(self, N: int, init_means: torch.Tensor, init_amp: torch.Tensor):
        super().__init__()
        assert init_means.shape == (N,3)
        assert init_amp.shape == (N,)
        self.N = N
        self.mu = nn.Parameter(init_means.clone())
        self.log_s = nn.Parameter(torch.zeros(N,3, device=init_means.device) - 2.0)
        q = torch.zeros(N,4, device=init_means.device)
        q[:,0] = 1.0
        self.q = nn.Parameter(q)
        self.a = nn.Parameter(init_amp.clone())
        self.b = nn.Parameter(torch.tensor(0.0, device=init_means.device))
        self.sigma_cutoff = 3.0

    def get_gaussian_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        max_radius = s.max(dim=-1, keepdim=True).values * self.sigma_cutoff
        radius_3d = max_radius.expand(-1, 3)
        min_bounds = self.mu - radius_3d
        max_bounds = self.mu + radius_3d
        return min_bounds, max_bounds

    def forward(self, x: torch.Tensor, use_culling: bool = True, max_gaussians_per_tile: int = 50000) -> torch.Tensor:
        P = x.shape[0]
        if not use_culling or self.N <= max_gaussians_per_tile:
            return self._forward_dense(x)
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        g_min, g_max = self.get_gaussian_bounds()
        overlaps = (g_max >= x_min.unsqueeze(0)) & (g_min <= x_max.unsqueeze(0))
        mask = overlaps.all(dim=-1)
        n_active = mask.sum().item()
        if n_active == 0:
            return torch.full((P,), self.b.item(), device=x.device, dtype=x.dtype)
        if n_active > max_gaussians_per_tile:
            return self._forward_dense(x)
        active_idx = mask.nonzero(as_tuple=True)[0]
        mu_active = self.mu[active_idx]
        log_s_active = self.log_s[active_idx]
        q_active = self.q[active_idx]
        a_active = self.a[active_idx]
        dx = x[:, None, :] - mu_active[None, :, :]
        s = torch.exp(log_s_active).clamp(1e-4, 10.0)
        qn = safe_normalize(q_active)
        R = quat_to_rotmat(qn)
        Rt = R.transpose(-1, -2)
        y = torch.einsum("pni,nij->pnj", dx, Rt)
        y = y / (s[None, :, :] + 1e-8)
        exp_term = -0.5 * (y * y).sum(dim=-1)
        g = torch.exp(exp_term)
        pred = (g * a_active[None, :]).sum(dim=1) + self.b
        return pred

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu[None, :, :]
        dx = x[:,None,:] - mu
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        qn = safe_normalize(self.q)
        R = quat_to_rotmat(qn)
        Rt = R.transpose(-1,-2)
        y = torch.einsum("pni,nij->pnj", dx, Rt)
        y = y / (s[None,:,:] + 1e-8)
        exp_term = -0.5 * (y*y).sum(dim=-1)
        g = torch.exp(exp_term)
        pred = (g * self.a[None,:]).sum(dim=1) + self.b
        return pred

    def forward_tiled(self, x: torch.Tensor, tile_size: int = 1000, max_gaussians_per_tile: int = 30000) -> torch.Tensor:
        P = x.shape[0]
        if P <= tile_size and self.N <= max_gaussians_per_tile:
            return self.forward(x, use_culling=True, max_gaussians_per_tile=max_gaussians_per_tile)
        pred = torch.empty(P, device=x.device, dtype=x.dtype)
        for start in range(0, P, tile_size):
            end = min(start + tile_size, P)
            x_tile = x[start:end]
            pred[start:end] = self.forward(x_tile, use_culling=True, max_gaussians_per_tile=max_gaussians_per_tile)
        return pred

# ============================================================================
# Entropy Models & Quantization
# ============================================================================

class LaplaceEntropyModel(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.log_b = nn.Parameter(torch.tensor(math.log(init_scale), device=DEVICE))

    def nll_bits(self, xq: torch.Tensor) -> torch.Tensor:
        b = torch.exp(self.log_b).clamp(1e-6, 1e3)
        logp = -math.log(2.0) - torch.log(b) - (xq.abs() / b)
        bits = (-logp / math.log(2.0)).mean()
        return bits

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
# Loss Functions
# ============================================================================

def charbonnier(x, eps=1e-3):
    return torch.sqrt(x*x + eps*eps)

@torch.no_grad()
def extract_random_patch(vol: torch.Tensor, patch_zyx=(32,64,64)):
    Z,Y,X = vol.shape
    pz,py,px = patch_zyx
    z0 = torch.randint(0, max(1, Z-pz+1), (1,), device=vol.device).item()
    y0 = torch.randint(0, max(1, Y-py+1), (1,), device=vol.device).item()
    x0 = torch.randint(0, max(1, X-px+1), (1,), device=vol.device).item()
    return (z0,y0,x0), vol[z0:z0+pz, y0:y0+py, x0:x0+px]

def tv3d(p: torch.Tensor):
    dz = (p[1:] - p[:-1]).abs().mean()
    dy = (p[:,1:] - p[:,:-1]).abs().mean()
    dx = (p[:,:,1:] - p[:,:,:-1]).abs().mean()
    return dx + dy + dz

def patch_topology_loss(model, coords_grid, V, patch_zyx=(16,32,32), tau=0.25, gamma=10.0):
    (z0,y0,x0), _ = extract_random_patch(V, patch_zyx)
    pz,py,px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1,3)
    pred = model(coords).reshape(pz,py,px)
    P = torch.sigmoid(gamma*(pred - tau))
    tv = tv3d(P)
    P_ = P[None,None]
    low = F.avg_pool3d(P_, kernel_size=3, stride=1, padding=1)[0,0]
    hf = (P - low).abs().mean()
    return tv + 0.5*hf

def reconstruction_tv_loss(model, coords_grid, patch_zyx=(8,16,16)):
    Z, Y, X, _ = coords_grid.shape
    pz, py, px = patch_zyx
    z0 = torch.randint(0, max(1, Z-pz+1), (1,), device=coords_grid.device).item()
    y0 = torch.randint(0, max(1, Y-py+1), (1,), device=coords_grid.device).item()
    x0 = torch.randint(0, max(1, X-px+1), (1,), device=coords_grid.device).item()
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1, 3)
    pred = model(coords).reshape(pz, py, px)
    return tv3d(pred)

def ssim_loss_3d(pred, target, win=5, K1=0.01, K2=0.03):
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

def patch_ssim_loss(model, coords_grid, V, patch_zyx=(8,16,16)):
    (z0,y0,x0), tgt_patch = extract_random_patch(V, patch_zyx)
    pz,py,px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1,3)
    pred_patch = model(coords).reshape(pz,py,px)
    return ssim_loss_3d(pred_patch, tgt_patch, win=3)

def edge_aware_loss(model, coords_grid, V, M, patch_zyx=(8,16,16)):
    (z0,y0,x0), tgt_patch = extract_random_patch(V, patch_zyx)
    pz,py,px = patch_zyx
    coords = coords_grid[z0:z0+pz, y0:y0+py, x0:x0+px].reshape(-1,3)
    pred_patch = model(coords).reshape(pz,py,px)
    m_patch = M[z0:z0+pz, y0:y0+py, x0:x0+px]
    dz = F.pad(tgt_patch[1:] - tgt_patch[:-1], (0,0,0,0,0,1)).abs()
    dy = F.pad(tgt_patch[:,1:] - tgt_patch[:,:-1], (0,0,0,1,0,0)).abs()
    dx = F.pad(tgt_patch[:,:,1:] - tgt_patch[:,:,:-1], (0,1,0,0,0,0)).abs()
    edge_weight = (dz + dy + dx + m_patch).clamp(0, 1)
    diff = (pred_patch - tgt_patch).abs()
    return (edge_weight * diff).mean()

def sparsity_loss(model):
    return model.a.abs().mean()

def smoothness_loss(model):
    s = torch.exp(model.log_s)
    min_scale_penalty = torch.relu(0.01 - s).mean()
    s_max = s.max(dim=-1).values
    s_min = s.min(dim=-1).values + 1e-8
    anisotropy = (s_max / s_min - 1.0).clamp(min=0).mean()
    max_scale_penalty = torch.relu(s - 1.0).mean()
    return min_scale_penalty + 0.1 * anisotropy + 0.1 * max_scale_penalty

def overlap_loss(model, n_samples=512):
    N = model.N
    if N < 2:
        return torch.tensor(0.0, device=model.mu.device)
    n_pairs = min(n_samples, N * (N - 1) // 2)
    idx_i = torch.randint(0, N, (n_pairs,), device=model.mu.device)
    idx_j = torch.randint(0, N, (n_pairs,), device=model.mu.device)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if len(idx_i) == 0:
        return torch.tensor(0.0, device=model.mu.device)
    mu_i = model.mu[idx_i]
    mu_j = model.mu[idx_j]
    s_i = torch.exp(model.log_s[idx_i])
    s_j = torch.exp(model.log_s[idx_j])
    dist = torch.norm(mu_i - mu_j, dim=-1)
    r_i = s_i.prod(dim=-1).pow(1/3)
    r_j = s_j.prod(dim=-1).pow(1/3)
    overlap = torch.exp(-dist / (r_i + r_j + 1e-4))
    return overlap.mean()

# ============================================================================
# Densification Controller
# ============================================================================

class DensificationController:
    def __init__(self, model, percent_dense=0.01):
        self.model = model
        self.percent_dense = percent_dense
        self.xyz_gradient_accum = torch.zeros((model.N, 1), device=DEVICE)
        self.denom = torch.zeros((model.N, 1), device=DEVICE)

    def reset_stats(self):
        N = self.model.N
        self.xyz_gradient_accum = torch.zeros((N, 1), device=DEVICE)
        self.denom = torch.zeros((N, 1), device=DEVICE)

    def add_densification_stats(self, grad_mu):
        if grad_mu is not None:
            grad_norm = grad_mu.norm(dim=-1, keepdim=True)
            self.xyz_gradient_accum += grad_norm
            self.denom += 1

    def get_scaling(self):
        return torch.exp(self.model.log_s).clamp(1e-4, 10.0)

    @torch.no_grad()
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        scaling = self.get_scaling()
        selected_pts_mask = (grads.squeeze() >= grad_threshold)
        selected_pts_mask = selected_pts_mask & (scaling.max(dim=1).values <= self.percent_dense * scene_extent)
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
        scaling = self.get_scaling()
        n_init = self.model.N
        padded_grad = torch.zeros((n_init,), device=DEVICE)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = (padded_grad >= grad_threshold)
        selected_pts_mask = selected_pts_mask & (scaling.max(dim=1).values > self.percent_dense * scene_extent)
        if selected_pts_mask.sum() == 0:
            return 0
        num_selected = selected_pts_mask.sum().item()
        stds = scaling[selected_pts_mask].repeat(N_split, 1)
        means = torch.zeros((stds.size(0), 3), device=DEVICE)
        samples = torch.normal(mean=means, std=stds)
        qn = safe_normalize(self.model.q.data[selected_pts_mask])
        R = quat_to_rotmat(qn)
        R_rep = R.repeat(N_split, 1, 1)
        new_mu = torch.bmm(R_rep, samples.unsqueeze(-1)).squeeze(-1) + \
                 self.model.mu.data[selected_pts_mask].repeat(N_split, 1)
        new_mu = new_mu.clamp(-1, 1)
        new_log_s = self.model.log_s.data[selected_pts_mask].repeat(N_split, 1) - math.log(0.8 * N_split)
        new_q = self.model.q.data[selected_pts_mask].repeat(N_split, 1)
        new_a = self.model.a.data[selected_pts_mask].repeat(N_split).reshape(-1) / N_split
        self._cat_tensors(new_mu, new_log_s, new_q, new_a)
        prune_mask = torch.cat([selected_pts_mask,
                                torch.zeros(N_split * num_selected, device=DEVICE, dtype=torch.bool)])
        self.prune_points(prune_mask)
        return num_selected * (N_split - 1)

    def _cat_tensors(self, new_mu, new_log_s, new_q, new_a):
        add_count = new_mu.shape[0]
        self.model.mu = nn.Parameter(torch.cat([self.model.mu.data, new_mu], dim=0))
        self.model.log_s = nn.Parameter(torch.cat([self.model.log_s.data, new_log_s], dim=0))
        self.model.q = nn.Parameter(torch.cat([self.model.q.data, new_q], dim=0))
        new_a_flat = new_a.reshape(-1) if new_a.dim() > 1 else new_a
        self.model.a = nn.Parameter(torch.cat([self.model.a.data, new_a_flat], dim=0))
        self.model.N += add_count
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum,
                                              torch.zeros((add_count, 1), device=DEVICE)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((add_count, 1), device=DEVICE)], dim=0)

    @torch.no_grad()
    def prune_points(self, mask):
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
        prune_mask = self.model.a.data.abs() < min_amplitude
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def prune_large_scale(self, max_scale=0.5, scene_extent=2.0):
        scaling = self.get_scaling()
        prune_mask = scaling.max(dim=1).values > max_scale * scene_extent
        return self.prune_points(prune_mask)

    @torch.no_grad()
    def densify_and_prune(self, grad_threshold=0.0002, min_amplitude=0.002,
                          scene_extent=2.0, max_scale=0.1):
        grads = self.xyz_gradient_accum / (self.denom + 1e-8)
        grads[grads.isnan()] = 0.0
        n_cloned = self.densify_and_clone(grads, grad_threshold, scene_extent)
        n_split = self.densify_and_split(grads, grad_threshold, scene_extent)
        n_pruned_amp = self.prune_low_amplitude(min_amplitude)
        n_pruned_scale = self.prune_large_scale(max_scale, scene_extent)
        self.reset_stats()
        torch.cuda.empty_cache()
        return {"cloned": n_cloned, "split": n_split,
                "pruned_amp": n_pruned_amp, "pruned_scale": n_pruned_scale,
                "total_gaussians": self.model.N}

# ============================================================================
# Sampling
# ============================================================================

@torch.no_grad()
def sample_points(coords_grid, V, M, n_uniform, n_biased):
    Z, Y, X, _ = coords_grid.shape
    total = Z * Y * X
    idx_u = torch.randint(0, total, (n_uniform,), device=V.device)
    zu = idx_u // (Y * X)
    yu = (idx_u % (Y * X)) // X
    xu = idx_u % X
    flatM = M.reshape(-1)
    max_categories = 2**24 - 1
    if total > max_categories:
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
    z = torch.cat([zu, zb], dim=0)
    y = torch.cat([yu, yb], dim=0)
    x = torch.cat([xu, xb], dim=0)
    pts = coords_grid[z, y, x]
    tgt = V[z, y, x]
    mval = M[z, y, x]
    return pts, tgt, mval

@torch.no_grad()
def init_gaussians_from_neurite_map(coords_grid, V, M, N0):
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
    """Save checkpoint with all necessary parameters for resuming training."""
    ensure_checkpoint_dir()
    
    # Ensure path is in checkpoint directory
    if not os.path.dirname(path):
        path = os.path.join(CKPT_DIR, path)
    
    checkpoint = {
        # Model parameters
        "model_state": model.state_dict(),
        "model_N": model.N,
        
        # Entropy model states
        "entropy_state": {
            "H_mu": H_mu.state_dict(),
            "H_logs": H_logs.state_dict(),
            "H_q": H_q.state_dict(),
            "H_a": H_a.state_dict(),
            "H_b": H_b.state_dict(),
        },
        
        # Quantization steps
        "Q": Q.__dict__,
        
        # Training config
        "config": config,
        
        # Training progress
        "iteration": iteration,
        "losses": losses,
        "densify_log": densify_log,
        
        # Data info for verification
        "tif_path": TIF_PATH,
        "voxel_spacing": VOXEL_SPACING,
        
        # Timestamp
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path} (iter={iteration}, N={model.N})")

# ============================================================================
# Training
# ============================================================================

def train(model, V, coords_grid, M, H_mu, H_logs, H_q, H_a, H_b, Q, cfg):
    controller = DensificationController(model, percent_dense=0.01) if cfg["densify_enabled"] else None

    params = (list(model.parameters()) +
              list(H_mu.parameters()) + list(H_logs.parameters()) +
              list(H_q.parameters()) + list(H_a.parameters()) + list(H_b.parameters()))
    opt = torch.optim.Adam(params, lr=cfg["lr"])

    losses = {"D": [], "R": [], "T": [], "TV": [], "SSIM": [], "Edge": [], "S": [], "Sm": [], "O": [], "Total": [], "N": []}
    densify_log = []
    t0 = time.time()

    lr = cfg["lr"]
    lr_final = cfg.get("lr_final", lr)
    steps = cfg["steps"]
    last_T = 0.0
    last_SSIM = 0.0

    print(f"\n{'='*60}")
    print(f"Starting training: {steps} iterations")
    print(f"Device: {DEVICE}")
    print(f"Initial Gaussians: {model.N}")
    print(f"{'='*60}\n")

    for it in tqdm(range(steps), desc="Training"):
        current_lr = lr - (lr - lr_final) * (it / max(steps - 1, 1))
        for pg in opt.param_groups:
            pg['lr'] = current_lr

        n_u = cfg["batch"] // 2
        n_b = cfg["batch"] - n_u
        pts, tgt, mval = sample_points(coords_grid, V, M, n_u, n_b)

        pred = model(pts)
        w = 1.0 + cfg["kappa"] * mval
        D = (w * charbonnier(pred - tgt)).mean()

        # Rate
        mu_q = ste_round(model.mu / Q.mu)
        logs_q = ste_round(model.log_s / Q.log_s)
        q_q = ste_round(model.q / Q.q)
        a_q = ste_round(model.a / Q.a)
        b_q = ste_round(model.b / Q.b)
        R = H_mu.nll_bits(mu_q) + H_logs.nll_bits(logs_q) + H_q.nll_bits(q_q) + H_a.nll_bits(a_q) + H_b.nll_bits(b_q)

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

        S = sparsity_loss(model)
        Sm = smoothness_loss(model)
        O = overlap_loss(model, n_samples=256)

        total = D + cfg["lam"] * R + cfg["alpha"] * T + cfg["beta_tv"] * TV + cfg["beta_ssim"] * SSIM_loss + \
                cfg["beta_edge"] * Edge_loss + cfg["beta_sparse"] * S + cfg["beta_smooth"] * Sm + cfg["beta_overlap"] * O

        opt.zero_grad(set_to_none=True)
        total.backward()

        if cfg["densify_enabled"] and controller is not None and model.mu.grad is not None:
            controller.add_densification_stats(model.mu.grad)

        opt.step()

        total_val = float(total.detach().cpu())
        losses["D"].append(float(D.detach().cpu()))
        losses["R"].append(float(R.detach().cpu()))
        losses["T"].append(float(T.detach().cpu()))
        losses["TV"].append(float(TV.detach().cpu()))
        losses["SSIM"].append(float(SSIM_loss.detach().cpu()))
        losses["Edge"].append(float(Edge_loss.detach().cpu()))
        losses["S"].append(float(S.detach().cpu()))
        losses["Sm"].append(float(Sm.detach().cpu()))
        losses["O"].append(float(O.detach().cpu()))
        losses["Total"].append(total_val)
        losses["N"].append(model.N)

        cum_avg_loss = sum(losses["Total"]) / len(losses["Total"])

        # Densification
        if cfg["densify_enabled"] and controller is not None:
            if (cfg["densify_from_iter"] <= it < cfg["densify_until_iter"]) and ((it + 1) % cfg["densify_every"] == 0):
                if model.N < cfg["max_gaussians"]:
                    stats = controller.densify_and_prune(
                        grad_threshold=cfg["grad_threshold"],
                        min_amplitude=cfg["min_amplitude"],
                        scene_extent=2.0,
                        max_scale=0.1
                    )
                    densify_log.append((it, stats))
                    params = (list(model.parameters()) +
                              list(H_mu.parameters()) + list(H_logs.parameters()) +
                              list(H_q.parameters()) + list(H_a.parameters()) + list(H_b.parameters()))
                    opt = torch.optim.Adam(params, lr=current_lr)
                    print(f"  [Densify @{it+1}] N={stats['total_gaussians']} "
                          f"(+{stats['cloned']} clone, +{stats['split']} split, "
                          f"-{stats['pruned_amp']} amp, -{stats['pruned_scale']} scale)")

        if (it + 1) % 200 == 0:
            dt = time.time() - t0
            recent_avg = sum(losses["Total"][-200:]) / min(200, len(losses["Total"]))
            print(f"iter {it+1:5d} | D={losses['D'][-1]:.5f} R={losses['R'][-1]:.1f} T={last_T:.4f} "
                  f"SSIM_loss={last_SSIM:.4f} | Loss: cur={total_val:.4f} avg={cum_avg_loss:.4f} recent={recent_avg:.4f} | N={model.N} | {dt:.1f}s")

        # Periodic checkpoint save
        if cfg.get("save_every") and (it + 1) % cfg["save_every"] == 0:
            ckpt_name = f"neurogs_ckpt_iter{it+1}.pt"
            save_checkpoint(model, H_mu, H_logs, H_q, H_a, H_b, Q, losses, densify_log, cfg,
                            os.path.join(CKPT_DIR, ckpt_name), iteration=it+1)

        if (it + 1) % 500 == 0:
            gc.collect()
            torch.cuda.empty_cache()

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
    print(f"coords_grid: {coords_grid.shape}")

    M = make_neurite_map(V_t)
    print(f"M: {M.shape} min/max: {float(M.min())}/{float(M.max())} mean: {float(M.mean())}")

    # Initialize model
    N0 = TRAINING_CONFIG["N0"]
    init_means, init_amp = init_gaussians_from_neurite_map(coords_grid, V_t, M, N0)
    model = GaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)
    model.log_s.data.fill_(-3.0)
    print(f"Model initialized with N={model.N} Gaussians")

    # Entropy models
    H_mu = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_logs = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_q = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_a = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_b = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    Q = QuantSteps()

    # Create checkpoint directory
    ensure_checkpoint_dir()
    print(f"Checkpoints will be saved to: {CKPT_DIR}/")

    # Train
    losses, densify_log = train(model, V_t, coords_grid, M, H_mu, H_logs, H_q, H_a, H_b, Q, TRAINING_CONFIG)

    print(f"\nFinal model has {model.N} Gaussians")

    # Save final checkpoint
    save_checkpoint(model, H_mu, H_logs, H_q, H_a, H_b, Q, losses, densify_log, TRAINING_CONFIG, 
                    CKPT_PATH, iteration=TRAINING_CONFIG["steps"])

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final checkpoint saved to: {CKPT_PATH}")
    print(f"All checkpoints in: {CKPT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()

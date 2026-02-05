#!/usr/bin/env python3
"""
Plot gradient-based edge vs flat error analysis for NeuroGS-Codec.

Generates a 2x3 figure:
  Row 1: Gradient Magnitude MIP | High-Gradient (Edge) Mask | Low-Gradient (Flat) Mask
  Row 2: Absolute Error MIP     | Error on EDGES            | Error on FLAT regions

Usage:
    python plot_edge_analysis.py
    python plot_edge_analysis.py --checkpoint path/to/checkpoint.pt --output edge_analysis.png
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import tifffile as tiff
import matplotlib.pyplot as plt
from typing import Tuple

# ============================================================================
# Model Definition
# ============================================================================

def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),   2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz, 2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),   ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

def safe_normalize(q: torch.Tensor, eps=1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)

class GaussianMixtureVolume(nn.Module):
    def __init__(self, N: int, init_means: torch.Tensor, init_amp: torch.Tensor):
        super().__init__()
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
        radius_3d = max_radius.expand(-1, 3)
        return self.mu - radius_3d, self.mu + radius_3d

    def forward(self, x: torch.Tensor, use_culling: bool = True, max_gaussians: int = 50000) -> torch.Tensor:
        P = x.shape[0]
        
        if not use_culling or self.N <= max_gaussians:
            return self._forward_dense(x)
        
        # Bounding box culling
        x_min, x_max = x.min(dim=0).values, x.max(dim=0).values
        g_min, g_max = self.get_gaussian_bounds()
        overlaps = (g_max >= x_min.unsqueeze(0)) & (g_min <= x_max.unsqueeze(0))
        mask = overlaps.all(dim=-1)
        n_active = mask.sum().item()
        
        if n_active == 0:
            return torch.full((P,), self.b.item(), device=x.device, dtype=x.dtype)
        if n_active > max_gaussians:
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
        return (g * a_active[None, :]).sum(dim=1) + self.b

    def _forward_dense(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.mu[None, :, :]
        dx = x[:, None, :] - mu
        s = torch.exp(self.log_s).clamp(1e-4, 10.0)
        qn = safe_normalize(self.q)
        R = quat_to_rotmat(qn)
        Rt = R.transpose(-1, -2)
        y = torch.einsum("pni,nij->pnj", dx, Rt)
        y = y / (s[None, :, :] + 1e-8)
        exp_term = -0.5 * (y * y).sum(dim=-1)
        g = torch.exp(exp_term)
        pred = (g * self.a[None, :]).sum(dim=1) + self.b
        return pred

# ============================================================================
# Utilities
# ============================================================================

def make_coord_grid_zyx(shape_zyx, spacing_xyz, device):
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

@torch.no_grad()
def reconstruct_full(model, coords_grid, tile_size=10000):
    Z, Y, X, _ = coords_grid.shape
    flat_coords = coords_grid.reshape(-1, 3)
    total = flat_coords.shape[0]
    pred = torch.empty(total, device=flat_coords.device)
    for start in range(0, total, tile_size):
        end = min(start + tile_size, total)
        pred[start:end] = model(flat_coords[start:end])
    V_hat = pred.reshape(Z, Y, X).clamp(0, 1)
    return V_hat

def compute_gradient_magnitude_3d(vol: torch.Tensor) -> torch.Tensor:
    """Compute gradient magnitude using finite differences."""
    D, H, W = vol.shape
    dz = torch.zeros_like(vol)
    dz[1:-1] = (vol[2:] - vol[:-2]) / 2.0
    dz[0] = vol[1] - vol[0]
    dz[-1] = vol[-1] - vol[-2]
    
    dy = torch.zeros_like(vol)
    dy[:, 1:-1] = (vol[:, 2:] - vol[:, :-2]) / 2.0
    dy[:, 0] = vol[:, 1] - vol[:, 0]
    dy[:, -1] = vol[:, -1] - vol[:, -2]
    
    dx = torch.zeros_like(vol)
    dx[:, :, 1:-1] = (vol[:, :, 2:] - vol[:, :, :-2]) / 2.0
    dx[:, :, 0] = vol[:, :, 1] - vol[:, :, 0]
    dx[:, :, -1] = vol[:, :, -1] - vol[:, :, -2]
    
    return torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)

def compute_gradient_edge_analysis(V_hat, V_t, threshold=0.01, percentile=90):
    """Compute gradient-based edge vs flat error analysis."""
    # Mask: only consider voxels where GT > threshold
    t = threshold * V_t.max()
    structure_mask = V_t > t
    
    # Compute gradient magnitude of GT
    grad_mag = compute_gradient_magnitude_3d(V_t)
    
    # Get gradient values only within structure
    grad_in_structure = grad_mag[structure_mask]
    
    # Compute percentile threshold for "high gradient" = edge
    # Use numpy for large tensors (torch.quantile has size limit)
    grad_np = grad_in_structure.cpu().numpy()
    grad_threshold = np.percentile(grad_np, percentile)
    grad_threshold = torch.tensor(grad_threshold, device=V_t.device)
    
    # Define edge and non-edge masks (within structure)
    high_grad_mask = (grad_mag > grad_threshold) & structure_mask
    low_grad_mask = (grad_mag <= grad_threshold) & structure_mask
    
    # Compute errors
    diff = (V_hat - V_t).abs()
    
    high_grad_count = high_grad_mask.sum().item()
    low_grad_count = low_grad_mask.sum().item()
    
    mae_high_grad = diff[high_grad_mask].mean().item() if high_grad_count > 0 else 0.0
    mae_low_grad = diff[low_grad_mask].mean().item() if low_grad_count > 0 else 0.0
    
    return {
        "grad_mag": grad_mag,
        "grad_threshold": grad_threshold.item(),
        "high_grad_mask": high_grad_mask,
        "low_grad_mask": low_grad_mask,
        "high_grad_count": high_grad_count,
        "low_grad_count": low_grad_count,
        "mae_high_grad": mae_high_grad,
        "mae_low_grad": mae_low_grad,
        "diff": diff,
    }

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot edge vs flat error analysis")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/neurogs_codec_ckpt_final.pt")
    parser.add_argument("--tif", type=str, default=None)
    parser.add_argument("--output", type=str, default="edge_analysis.png")
    parser.add_argument("--percentile", type=float, default=90.0, help="Percentile for edge threshold (default 90 = top 10%)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    N = ckpt["model_N"]
    tif_path = args.tif or ckpt.get("tif_path", "./10-2900-control-cell-05_cropped_corrected.tif")
    voxel_spacing = ckpt.get("voxel_spacing", (0.126, 0.126, 1.0))
    iteration = ckpt.get("iteration", "?")
    
    print(f"  Gaussians: {N}, Iteration: {iteration}")

    # Load ground truth
    print(f"Loading ground truth: {tif_path}")
    V_np = tiff.imread(tif_path)
    V = V_np.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    V_t = torch.from_numpy(V).to(device)
    print(f"  Volume shape: {V_t.shape}")

    # Build coordinate grid
    coords_grid = make_coord_grid_zyx(V_t.shape, voxel_spacing, device)

    # Create and load model
    dummy_means = torch.zeros(N, 3, device=device)
    dummy_amp = torch.zeros(N, device=device)
    model = GaussianMixtureVolume(N, dummy_means, dummy_amp).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Reconstruct
    print("Reconstructing volume...")
    V_hat = reconstruct_full(model, coords_grid)

    # Compute edge analysis
    print(f"Computing gradient-based edge analysis (percentile={args.percentile})...")
    results = compute_gradient_edge_analysis(V_hat, V_t, threshold=0.01, percentile=args.percentile)

    print(f"  Edge voxels: {results['high_grad_count']:,} (top {100-args.percentile:.0f}%)")
    print(f"  Flat voxels: {results['low_grad_count']:,}")
    print(f"  Edge MAE: {results['mae_high_grad']:.5f}")
    print(f"  Flat MAE: {results['mae_low_grad']:.5f}")
    print(f"  Ratio (edge/flat): {results['mae_high_grad']/(results['mae_low_grad']+1e-8):.2f}x")

    # Create visualization
    print("Generating plot...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Gradient analysis
    # Gradient Magnitude MIP
    grad_mip = results['grad_mag'].max(dim=0).values.cpu().numpy()
    im00 = axes[0, 0].imshow(grad_mip, cmap='viridis')
    axes[0, 0].set_title(f'Gradient Magnitude MIP\nthreshold={results["grad_threshold"]:.4f}', fontsize=11)
    axes[0, 0].axis('off')
    plt.colorbar(im00, ax=axes[0, 0], fraction=0.046)

    # High-Gradient (Edge) Mask
    high_grad_mip = results['high_grad_mask'].max(dim=0).values.cpu().numpy()
    axes[0, 1].imshow(high_grad_mip, cmap='Reds')
    axes[0, 1].set_title(f'High-Gradient (Edge) Mask\n{results["high_grad_count"]:,} voxels (top {100-args.percentile:.0f}%)', fontsize=11)
    axes[0, 1].axis('off')

    # Low-Gradient (Flat) Mask
    low_grad_mip = results['low_grad_mask'].max(dim=0).values.cpu().numpy()
    axes[0, 2].imshow(low_grad_mip, cmap='Blues')
    axes[0, 2].set_title(f'Low-Gradient (Flat) Mask\n{results["low_grad_count"]:,} voxels', fontsize=11)
    axes[0, 2].axis('off')

    # Row 2: Error analysis
    # Absolute Error MIP
    diff_mip = results['diff'].max(dim=0).values.cpu().numpy()
    im10 = axes[1, 0].imshow(diff_mip, cmap='inferno')
    axes[1, 0].set_title('Absolute Error MIP', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im10, ax=axes[1, 0], fraction=0.046)

    # Error on EDGES
    error_high = torch.zeros_like(results['diff'])
    error_high[results['high_grad_mask']] = results['diff'][results['high_grad_mask']]
    error_high_mip = error_high.max(dim=0).values.cpu().numpy()
    axes[1, 1].imshow(error_high_mip, cmap='Reds')
    axes[1, 1].set_title(f'Error on EDGES\nMAE={results["mae_high_grad"]:.5f}', fontsize=11)
    axes[1, 1].axis('off')

    # Error on FLAT regions
    error_low = torch.zeros_like(results['diff'])
    error_low[results['low_grad_mask']] = results['diff'][results['low_grad_mask']]
    error_low_mip = error_low.max(dim=0).values.cpu().numpy()
    axes[1, 2].imshow(error_low_mip, cmap='Blues')
    axes[1, 2].set_title(f'Error on FLAT regions\nMAE={results["mae_low_grad"]:.5f}', fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()

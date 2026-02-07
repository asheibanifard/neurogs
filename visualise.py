#!/usr/bin/env python3
"""
Visualise NeuroGS-Codec checkpoint: XY projection (MIP along Z).

Usage:
    python visualise.py
    python visualise.py --checkpoint path/to/checkpoint.pt
    python visualise.py --checkpoint neurogs_codec_ckpt_final.pt --tif path/to/volume.tif
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
# Model Definition (must match training code)
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
# Coordinate Grid
# ============================================================================

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
# Reconstruction
# ============================================================================

@torch.no_grad()
def reconstruct_full(model, coords_grid, tile_size=10000):
    """Reconstruct full volume from model."""
    Z, Y, X, _ = coords_grid.shape
    flat_coords = coords_grid.reshape(-1, 3)
    total = flat_coords.shape[0]
    pred = torch.empty(total, device=flat_coords.device)
    
    for start in range(0, total, tile_size):
        end = min(start + tile_size, total)
        pred[start:end] = model(flat_coords[start:end])
    
    V_hat = pred.reshape(Z, Y, X).clamp(0, 1)
    return V_hat

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualise NeuroGS-Codec checkpoint")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/neurogs_codec_ckpt_final.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--tif", type=str, default=None,
                        help="Path to ground truth TIFF (overrides checkpoint metadata)")
    parser.add_argument("--output", type=str, default='xy_projection2.png',
                        help="Save figure to file instead of displaying")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    # Get metadata from checkpoint
    N = ckpt["model_N"]
    tif_path = args.tif or ckpt.get("tif_path", "./10-2900-control-cell-05_cropped_corrected.tif")
    voxel_spacing = ckpt.get("voxel_spacing", (0.126, 0.126, 1.0))
    iteration = ckpt.get("iteration", "?")
    
    print(f"  Gaussians: {N}")
    print(f"  Iteration: {iteration}")
    print(f"  TIF: {tif_path}")
    print(f"  Voxel spacing: {voxel_spacing}")

    # Load ground truth volume
    print(f"Loading ground truth: {tif_path}")
    V_np = tiff.imread(tif_path)
    V = V_np.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    V_t = torch.from_numpy(V).to(device)
    print(f"  Volume shape: {V_t.shape}")

    # Build coordinate grid
    coords_grid = make_coord_grid_zyx(V_t.shape, voxel_spacing, device)

    # Create model and load state
    dummy_means = torch.zeros(N, 3, device=device)
    dummy_amp = torch.zeros(N, device=device)
    model = GaussianMixtureVolume(N, dummy_means, dummy_amp).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded successfully")

    # Reconstruct
    print("Reconstructing volume...")
    V_hat = reconstruct_full(model, coords_grid)
    print(f"  Reconstruction shape: {V_hat.shape}")

    # Compute metrics
    mse = ((V_hat - V_t) ** 2).mean().item()
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    mae = (V_hat - V_t).abs().mean().item()
    print(f"\nMetrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  MAE:  {mae:.6f}")

    # Visualize XY projection (MIP along Z)
    print("\nGenerating XY projection visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth XY projection
    gt_xy = V_t.max(dim=0).values.cpu().numpy()
    im0 = axes[0].imshow(gt_xy, cmap='gray')
    axes[0].set_title('Ground Truth (XY MIP)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Reconstruction XY projection
    recon_xy = V_hat.max(dim=0).values.cpu().numpy()
    im1 = axes[1].imshow(recon_xy, cmap='gray')
    axes[1].set_title('Reconstruction (XY MIP)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Difference
    diff_xy = (V_hat - V_t).abs().max(dim=0).values.cpu().numpy()
    im2 = axes[2].imshow(diff_xy, cmap='inferno')
    axes[2].set_title('Absolute Difference (XY MIP)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.suptitle(f"NeuroGS-Codec | N={N} | Iter={iteration} | PSNR={psnr:.2f}dB", fontsize=12)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {args.output}")
    else:
        plt.show()

    print("\nDone!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick GPU timing check"""

import time
import torch
import numpy as np
import tifffile as tiff
import yaml
import sys

# Prevent neurogs_v7 from running training
sys.argv = ["quick_profile.py"]

from neurogs_v7 import (
    GaussianMixtureField,
    sample_points_with_neighbors,
    loss_volume,
    HAS_CUDA_EXTENSION,
)

print(f"CUDA extension loaded: {HAS_CUDA_EXTENSION}")

# Load minimal data
cfg = yaml.safe_load(open("config.yml"))
vol = tiff.imread(cfg["data"]["tif_path"]).astype(np.float32)
vmin, vmax = vol.min(), vol.max()
vol = (vol - vmin) / (vmax - vmin)

device = "cuda"
vol_gpu = torch.from_numpy(vol).float().to(device)

# Create small model
field = GaussianMixtureField(
    num_gaussians=10000,
    init_scale=0.09,
    init_amplitude=0.05,
    bounds=None,
).to(device)

# Sample points
vol_pts = 4096
x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
    vol_gpu, vol_pts, delta_vox=1, intensity_weighted=True
)

# Warmup
for _ in range(3):
    lv, _ = loss_volume(field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz, 
                        w_grad=0.3, w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05)
    lv.backward()
torch.cuda.synchronize()

# Time forward
times_fwd = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    lv, _ = loss_volume(field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                        w_grad=0.3, w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05)
    torch.cuda.synchronize()
    times_fwd.append((time.time() - t0) * 1000)

# Time backward
optimizer = torch.optim.Adam(field.parameters())
times_bwd = []
for _ in range(10):
    optimizer.zero_grad()
    lv, _ = loss_volume(field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
                        w_grad=0.3, w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05)
    
    torch.cuda.synchronize()
    t0 = time.time()
    lv.backward()
    torch.cuda.synchronize()
    times_bwd.append((time.time() - t0) * 1000)

print(f"\nK={field.num_gaussians}, N={vol_pts}")
print(f"Forward:  {np.mean(times_fwd):.1f} ± {np.std(times_fwd):.1f} ms")
print(f"Backward: {np.mean(times_bwd):.1f} ± {np.std(times_bwd):.1f} ms")
print(f"Total:    {np.mean(times_fwd) + np.mean(times_bwd):.1f} ms/step ({1000/(np.mean(times_fwd) + np.mean(times_bwd)):.1f} it/s)")

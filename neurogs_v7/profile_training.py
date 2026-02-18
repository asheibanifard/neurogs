#!/usr/bin/env python3
"""Quick profiling script to identify training bottlenecks."""

import time
import torch
import numpy as np
import tifffile as tiff
import yaml

from neurogs_v7 import (
    GaussianMixtureField,
    sample_points_with_neighbors,
    sample_points_from_volume,
    loss_volume,
    HAS_CUDA_EXTENSION,
)

# Load config and data
cfg = yaml.safe_load(open("config.yml"))
vol = tiff.imread(cfg["data"]["tif_path"]).astype(np.float32)
vmin, vmax = vol.min(), vol.max()
vol = (vol - vmin) / (vmax - vmin)

device = "cuda"
vol_gpu = torch.from_numpy(vol).float().to(device)

# Create model
mc = cfg["model"]
field = GaussianMixtureField(
    num_gaussians=int(mc["num_gaussians"]),
    init_scale=float(mc.get("init_scale", 0.05)),
    init_amplitude=float(mc.get("init_amplitude", 0.1)),
    bounds=mc.get("bounds"),
).to(device)

tc = cfg["training"]
vol_pts = int(tc.get("vol_points_per_step", 8192))
use_grad = bool(tc.get("use_grad_loss", True))
delta_vox = int(tc.get("grad_delta_vox", 1))
w_grad = float(tc.get("lambda_grad", 0.3))

# Warmup
for _ in range(5):
    if use_grad:
        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
            vol_gpu, vol_pts, delta_vox=delta_vox, intensity_weighted=True
        )
    else:
        x, v = sample_points_from_volume(vol_gpu, vol_pts, intensity_weighted=True)
        x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None
    _ = loss_volume(field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz, w_grad=w_grad)

torch.cuda.synchronize()

# Profile components
n_iter = 20
print(f"\nProfiling {n_iter} iterations with:")
print(f"  K={field.num_gaussians} Gaussians")
print(f"  N={vol_pts} points per step")
print(f"  Gradient supervision: {use_grad}")
print(f"  CUDA extension: {HAS_CUDA_EXTENSION}")
print("=" * 60)

# 1. Sampling
times_sample = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    t0 = time.time()
    if use_grad:
        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
            vol_gpu, vol_pts, delta_vox=delta_vox, intensity_weighted=True, cache_key="vol"
        )
    else:
        x, v = sample_points_from_volume(vol_gpu, vol_pts, intensity_weighted=True, cache_key="vol")
        x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None
    torch.cuda.synchronize()
    times_sample.append(time.time() - t0)

# 2. Forward pass (volume loss)
times_forward = []
for _ in range(n_iter):
    if use_grad:
        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
            vol_gpu, vol_pts, delta_vox=delta_vox, intensity_weighted=True, cache_key="vol"
        )
    else:
        x, v = sample_points_from_volume(vol_gpu, vol_pts, intensity_weighted=True, cache_key="vol")
        x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None
    
    torch.cuda.synchronize()
    t0 = time.time()
    lv, pv = loss_volume(
        field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
        w_grad=w_grad if use_grad else 0.0,
        w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05,
    )
    torch.cuda.synchronize()
    times_forward.append(time.time() - t0)

# 3. Backward pass
optimizer = torch.optim.Adam(field.parameters(), lr=1e-3)
times_backward = []
for _ in range(n_iter):
    if use_grad:
        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
            vol_gpu, vol_pts, delta_vox=delta_vox, intensity_weighted=True, cache_key="vol"
        )
    else:
        x, v = sample_points_from_volume(vol_gpu, vol_pts, intensity_weighted=True, cache_key="vol")
        x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None
    
    optimizer.zero_grad()
    lv, pv = loss_volume(
        field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
        w_grad=w_grad if use_grad else 0.0,
        w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05,
    )
    
    torch.cuda.synchronize()
    t0 = time.time()
    lv.backward()
    torch.cuda.synchronize()
    times_backward.append(time.time() - t0)

# 4. Optimizer step
times_optim = []
for _ in range(n_iter):
    if use_grad:
        x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz = sample_points_with_neighbors(
            vol_gpu, vol_pts, delta_vox=delta_vox, intensity_weighted=True, cache_key="vol"
        )
    else:
        x, v = sample_points_from_volume(vol_gpu, vol_pts, intensity_weighted=True, cache_key="vol")
        x_dx = v_dx = x_dy = v_dy = x_dz = v_dz = None
    
    optimizer.zero_grad()
    lv, pv = loss_volume(
        field, x, v, x_dx, v_dx, x_dy, v_dy, x_dz, v_dz,
        w_grad=w_grad if use_grad else 0.0,
        w_tube=1e-4, w_cross=1e-4, w_scale=5e-4, scale_target=0.05,
    )
    lv.backward()
    
    torch.cuda.synchronize()
    t0 = time.time()
    optimizer.step()
    torch.cuda.synchronize()
    times_optim.append(time.time() - t0)

# Report
ts = np.array(times_sample) * 1000
tf = np.array(times_forward) * 1000
tb = np.array(times_backward) * 1000
to = np.array(times_optim) * 1000

print(f"\nComponent            Mean ± Std (ms)    % of Total")
print("=" * 60)
total_mean = ts.mean() + tf.mean() + tb.mean() + to.mean()
print(f"Sampling           {ts.mean():7.2f} ± {ts.std():5.2f}      {100*ts.mean()/total_mean:5.1f}%")
print(f"Forward (loss)     {tf.mean():7.2f} ± {tf.std():5.2f}      {100*tf.mean()/total_mean:5.1f}%")
print(f"Backward           {tb.mean():7.2f} ± {tb.std():5.2f}      {100*tb.mean()/total_mean:5.1f}%")
print(f"Optimizer step     {to.mean():7.2f} ± {to.std():5.2f}      {100*to.mean()/total_mean:5.1f}%")
print("-" * 60)
print(f"TOTAL per step     {total_mean:7.2f} ms             ({1000/total_mean:.1f} it/s)")
print("=" * 60)

# Breakdown of forward pass
if use_grad:
    print("\nForward pass breakdown:")
    print("  - Field evaluation on center: 1x forward")
    print("  - Field evaluation on neighbors: 3x forward (x_dx, x_dy, x_dz)")
    print("  - Total evaluations: 4x field forward passes")
    print("  - Gradient loss is the bottleneck if enabled")

#!/usr/bin/env python3
"""
CUDA-accelerated Gaussian splatting training with regularization + pruning.

Uses a custom CUDA kernel for the alpha-compositing splatting inner loop,
avoiding the (N_pixels × K_gaussians) intermediate tensor that causes OOM.

Pipeline per training step:
  PyTorch autograd: params → covariance → camera transform → 2D projection → cov_inv
  CUDA kernel:      cov_inv + means_2d + opacities → rendered pixels (no N×K tensor)
  PyTorch autograd: loss.backward() chains through CUDA backward + projection

Regularization:
  - Opacity entropy: pushes dim Gaussians toward 0 (reduces noise)
  - Scale penalty: penalises too-small Gaussians
  - Periodic pruning of low-opacity Gaussians
"""
import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent dir for rendering module, and neurogs_v7 root for splat_cuda
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from rendering.rendering import (
    Camera, compute_aspect_scales,
    orbit_camera_pose, generate_camera_poses, generate_mip_dataset, load_volume,
)
from splat_cuda_wrapper import CUDASplattingTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── 1. Load volume for MIP ground truth ──────────────────────────
vol_path = os.path.join(BASE_DIR,
                        "10-2900-control-cell-05_cropped_corrected.tif")
vol_path = os.path.abspath(vol_path)
print(f"Loading volume: {vol_path}")
vol_np = load_volume(vol_path)
Z, Y, X = vol_np.shape
print(f"  Volume shape (Z,Y,X): ({Z}, {Y}, {X})")
vol_gpu = torch.from_numpy(vol_np).to(device)

aspect_scales = compute_aspect_scales((Z, Y, X))
print(f"  Aspect scales: {aspect_scales.tolist()}")

# ── 2. Load checkpoint ───────────────────────────────────────────
ckpt_path = os.path.join(BASE_DIR, "checkpoints", "gmf_refined_best.pt")
ckpt_path = os.path.abspath(ckpt_path)
print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

means = ckpt["means"].to(device)
log_scales = ckpt["log_scales"].to(device)
quaternions = ckpt["quaternions"].to(device)
log_amplitudes = ckpt["log_amplitudes"].to(device)
K = means.shape[0]
print(f"  {K} Gaussians loaded")

# ── 3. Camera & MIP dataset ──────────────────────────────────────
H, W = 256, 256
camera = Camera.from_fov(fov_x_deg=50.0, width=W, height=H,
                         near=0.01, far=10.0)

poses = generate_camera_poses(
    n_azimuth=12, n_elevation=5,
    elevation_range=(-60.0, 60.0),
    radius=3.5,
    include_axis_aligned=True,
)
print(f"  {len(poses)} camera poses")

print("Rendering MIP ground truth...")
dataset = generate_mip_dataset(
    vol_gpu, camera, poses,
    n_ray_samples=200, near=0.5, far=6.0,
    aspect_scales=aspect_scales,
)
print(f"  {len(dataset)} views, {H}x{W}")

# Convert dataset to dicts with camera intrinsics for the CUDA trainer
camera_dicts = []
for view in dataset:
    camera_dicts.append({
        'R': view['R'].to(device),
        'T': view['T'].to(device),
        'image': view['image'].to(device),
        'fx': camera.fx,
        'fy': camera.fy,
        'cx': camera.cx,
        'cy': camera.cy,
        'width': camera.width,
        'height': camera.height,
    })

# ── 4. CUDA-accelerated training with regularization ─────────────
print("\n=== CUDA Splatting Training with Regularization ===")
trainer = CUDASplattingTrainer(
    means=means,
    log_scales=log_scales,
    quaternions=quaternions,
    log_amplitudes=log_amplitudes,
    aspect_scales=aspect_scales,
    lr=5e-4,
    pixels_per_step=16384,  # 2× more pixels per step thanks to CUDA speed
)

# Regularization strengths
trainer.lambda_opacity = 0.01
trainer.lambda_scale = 0.001
trainer.scale_min_target = 0.005
trainer.prune_every = 2000
trainer.prune_opacity_thresh = 0.01
trainer.prune_min_gaussians = 2000

save_template = os.path.join(BASE_DIR, "checkpoints", "splat_clean_step{step}.pt")
n_steps = 20000
log_every = 200
save_every = 2000
n_views = len(camera_dicts)

print(f"  {K} Gaussians, {n_views} views, {trainer.pixels_per_step} pixels/step")
print(f"  Regularization: lambda_opacity={trainer.lambda_opacity}, lambda_scale={trainer.lambda_scale}")
print("-" * 60)

history = []
t0 = time.time()

for step in range(1, n_steps + 1):
    # Periodic pruning
    if trainer.prune_every > 0 and step % trainer.prune_every == 0:
        trainer.prune_gaussians(step)

    # Random view
    view_idx = torch.randint(0, n_views, (1,)).item()
    view = camera_dicts[view_idx]

    metrics = trainer.train_step(view, view['image'])
    history.append(metrics)

    if step % log_every == 0:
        avg_loss = np.mean([h['loss'] for h in history[-log_every:]])
        avg_l1 = np.mean([h['l1'] for h in history[-log_every:]])
        n_gauss = trainer.means.shape[0]
        elapsed_so_far = time.time() - t0
        steps_per_sec = step / elapsed_so_far
        print(f"  Step {step:>6d}/{n_steps}  |  loss={avg_loss:.6f}  "
              f"l1={avg_l1:.6f}  K={n_gauss}  ({steps_per_sec:.1f} it/s)")

    if step % save_every == 0:
        path = save_template.format(step=step)
        trainer.save_checkpoint(path, step)
        print(f"  Checkpoint -> {path}")

elapsed = time.time() - t0
final_K = trainer.means.shape[0]
print(f"\nTraining complete: {elapsed:.0f}s ({elapsed/n_steps*1000:.1f}ms/step), "
      f"{K} -> {final_K} Gaussians")

# Save final
best_path = os.path.join(BASE_DIR, "checkpoints", "splat_clean_best.pt")
trainer.save_checkpoint(best_path, n_steps)
print(f"Saved -> {best_path}")

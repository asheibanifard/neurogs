"""
CUDA-accelerated Gaussian Splatting for Training
=================================================

Fuses the N×K inner splatting loop in CUDA while using PyTorch autograd
for the projection chain (quaternion→rotation→covariance→camera→2D).

Architecture:
  PyTorch autograd:  means, log_scales, quaternions, log_amplitudes
                     → covariances, opacities
                     → camera transform → 2D means, 2D covariance
                     → invert 2×2 → cov_inv (K,3)
  CUDA kernel:       cov_inv, means_2d, opacities, colors, pixels
                     → rendered (N,)   [forward]
                     → grad_means_2d, grad_cov_inv, grad_opacities, grad_colors [backward]
  PyTorch autograd:  chain rule back through projection to learnable params

This avoids materialising the (N, K) Gaussian evaluation tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Tuple, Optional

try:
    import splat_cuda
    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False
    print("WARNING: splat_cuda not found. Build with: python setup_splat_cuda.py build_ext --inplace")


class SplatAlphaFunction(Function):
    """
    Custom autograd function wrapping the CUDA splatting kernel.

    Forward: (means_2d, cov_inv, opacities, colors, pixels) → rendered
    Backward: grad_rendered → (grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None)
    """

    @staticmethod
    def forward(ctx, means_2d, cov_inv, opacities, colors, pixels):
        """
        Parameters
        ----------
        means_2d : (K, 2) float32 — sorted by depth
        cov_inv : (K, 3) float32 — [inv_a, inv_b, inv_d] sorted
        opacities : (K,) float32 — sorted
        colors : (K,) float32 — sorted (grayscale)
        pixels : (N, 2) float32

        Returns
        -------
        rendered : (N,) float32
        """
        rendered, T_out = splat_cuda.forward(
            means_2d, cov_inv, opacities, colors, pixels
        )
        ctx.save_for_backward(means_2d, cov_inv, opacities, colors, pixels)
        return rendered

    @staticmethod
    def backward(ctx, grad_rendered):
        means_2d, cov_inv, opacities, colors, pixels = ctx.saved_tensors

        grad_means_2d, grad_cov_inv, grad_opacities, grad_colors = splat_cuda.backward(
            grad_rendered.contiguous(),
            means_2d, cov_inv, opacities, colors, pixels
        )

        return grad_means_2d, grad_cov_inv, grad_opacities, grad_colors, None


def cuda_splat_alpha(means_2d, cov_inv, opacities, colors, pixels):
    """Differentiable alpha-compositing splatting via CUDA."""
    return SplatAlphaFunction.apply(means_2d, cov_inv, opacities, colors, pixels)


# ============================================================================
# Full differentiable pipeline: learnable params → rendered pixels
# ============================================================================

def build_covariances(quaternions, log_scales):
    """
    Build 3D covariance matrices from quaternions and log-scales.

    Parameters
    ----------
    quaternions : (K, 4)
    log_scales : (K, 3)

    Returns
    -------
    covariances : (K, 3, 3)
    """
    K = quaternions.shape[0]

    scales = torch.exp(log_scales).clamp(1e-5, 1e2)
    q = F.normalize(quaternions, p=2, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(K, 3, 3, device=q.device, dtype=q.dtype)
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)

    S2 = torch.diag_embed(scales ** 2)
    covariances = R @ S2 @ R.transpose(-2, -1)

    return covariances


def transform_to_camera(means, covariances, R_cam, T_cam):
    """Transform Gaussians from world to camera frame."""
    means_cam = (R_cam @ means.unsqueeze(-1)).squeeze(-1) + T_cam.unsqueeze(0)
    cov_cam = R_cam.unsqueeze(0) @ covariances @ R_cam.T.unsqueeze(0)
    return means_cam, cov_cam


def project_to_2d(means_cam, cov_cam, fx, fy, cx, cy):
    """
    Project 3D Gaussians to 2D.

    Returns means_2d (K,2), cov_2d (K,2,2), depths (K,).
    """
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2]
    z_safe = z.clamp(min=1e-6)

    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy
    means_2d = torch.stack([u, v], dim=-1)

    # Jacobian
    z_sq = z_safe * z_safe
    K_g = means_cam.shape[0]
    J = torch.zeros(K_g, 2, 3, device=means_cam.device, dtype=means_cam.dtype)
    J[:, 0, 0] = fx / z_safe
    J[:, 0, 2] = -fx * x / z_sq
    J[:, 1, 1] = fy / z_safe
    J[:, 1, 2] = -fy * y / z_sq

    cov_2d = J @ cov_cam @ J.transpose(-2, -1)
    eps = 1e-4
    cov_2d = cov_2d + eps * torch.eye(2, device=cov_2d.device).unsqueeze(0)

    return means_2d, cov_2d, z


def invert_2x2(cov_2d):
    """
    Invert 2×2 symmetric matrices.

    Parameters
    ----------
    cov_2d : (K, 2, 2)

    Returns
    -------
    cov_inv_packed : (K, 3) — [inv_a, inv_b, inv_d]
    """
    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]

    det = a * d - b * b
    det_safe = det.clamp(min=1e-12)
    inv_det = 1.0 / det_safe

    inv_a = d * inv_det
    inv_b = -b * inv_det
    inv_d = a * inv_det

    return torch.stack([inv_a, inv_b, inv_d], dim=-1)


def cull_gaussians(means_cam, means_2d, cov_2d, near, far, width, height, radius_mult=3.0):
    """Visibility culling: depth + frustum."""
    z = means_cam[:, 2]
    depth_ok = (z > near) & (z < far)

    a = cov_2d[:, 0, 0]
    b = cov_2d[:, 0, 1]
    d = cov_2d[:, 1, 1]
    tr = a + d
    det = a * d - b * b
    disc = (tr * tr - 4.0 * det).clamp(min=0.0)
    lambda_max = 0.5 * (tr + torch.sqrt(disc))
    radius = radius_mult * torch.sqrt(lambda_max.clamp(min=1e-8))

    u = means_2d[:, 0]
    v = means_2d[:, 1]
    in_frustum = (
        (u + radius > 0) & (u - radius < width) &
        (v + radius > 0) & (v - radius < height)
    )

    return depth_ok & in_frustum


def apply_aspect_correction(means, covariances, aspect_scales):
    """Scale means and covariances by aspect ratios."""
    s = aspect_scales.to(means.device)
    means_c = means * s.unsqueeze(0)
    S = torch.diag(s)
    cov_c = S.unsqueeze(0) @ covariances @ S.unsqueeze(0)
    return means_c, cov_c


class CUDASplattingRenderer:
    """
    CUDA-accelerated Gaussian splatting renderer for training.

    Uses PyTorch autograd for projection chain + CUDA kernel for splatting.
    """

    def __init__(self, near=0.01, far=10.0, radius_mult=3.0):
        self.near = near
        self.far = far
        self.radius_mult = radius_mult
        assert HAS_CUDA, "splat_cuda extension required. Build with setup_splat_cuda.py"

    def render_at_pixels(
        self,
        means, covariances, opacities, colors,
        R_cam, T_cam,
        fx, fy, cx, cy, width, height,
        pixels,
    ):
        """
        Render at specific pixel locations using CUDA splatting.

        Parameters
        ----------
        means : (K, 3) - world space (aspect-corrected)
        covariances : (K, 3, 3)
        opacities : (K,) - differentiable
        colors : (K,) - grayscale color (differentiable)
        R_cam, T_cam : camera extrinsics
        fx, fy, cx, cy : camera intrinsics
        width, height : image dimensions
        pixels : (N, 2) pixel coordinates

        Returns
        -------
        rendered : (N,) - differentiable w.r.t. all inputs
        """
        # Step 1: World → Camera (differentiable via autograd)
        means_cam, cov_cam = transform_to_camera(means, covariances, R_cam, T_cam)

        # Step 2: 3D → 2D projection (differentiable via autograd)
        means_2d, cov_2d, depths = project_to_2d(means_cam, cov_cam, fx, fy, cx, cy)

        # Visibility culling (no grad)
        with torch.no_grad():
            visible = cull_gaussians(
                means_cam, means_2d, cov_2d,
                self.near, self.far, width, height, self.radius_mult
            )

        n_vis = visible.sum().item()
        if n_vis == 0:
            return opacities.sum() * 0.0 + torch.zeros(
                pixels.shape[0], device=pixels.device)

        # Filter to visible
        means_2d_vis = means_2d[visible]
        cov_2d_vis = cov_2d[visible]
        opacities_vis = opacities[visible]
        colors_vis = colors[visible]
        depths_vis = depths[visible]

        # Sort by depth (differentiable: just reordering)
        order = torch.argsort(depths_vis)
        means_2d_sorted = means_2d_vis[order]
        cov_2d_sorted = cov_2d_vis[order]
        opacities_sorted = opacities_vis[order]
        colors_sorted = colors_vis[order]

        # Invert 2×2 covariance (differentiable via autograd)
        cov_inv = invert_2x2(cov_2d_sorted)  # (K_vis, 3)

        # Step 3+4: CUDA splatting (custom autograd backward)
        rendered = cuda_splat_alpha(
            means_2d_sorted, cov_inv, opacities_sorted, colors_sorted, pixels
        )

        return rendered


class CUDASplattingTrainer:
    """
    Training loop with CUDA-accelerated splatting + opacity/scale regularization + pruning.
    """

    def __init__(
        self,
        means, log_scales, quaternions, log_amplitudes,
        aspect_scales,
        lr=5e-4,
        pixels_per_step=16384,
    ):
        self.device = means.device
        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.quaternions = nn.Parameter(quaternions.clone())
        self.log_amplitudes = nn.Parameter(log_amplitudes.clone())
        self.aspect_scales = aspect_scales.to(self.device)

        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': lr},
            {'params': [self.log_scales], 'lr': lr * 0.5},
            {'params': [self.quaternions], 'lr': lr * 0.3},
            {'params': [self.log_amplitudes], 'lr': lr},
        ])

        self.pixels_per_step = pixels_per_step
        self.renderer = CUDASplattingRenderer()

        # Regularization
        self.lambda_opacity = 0.01
        self.lambda_scale = 0.001
        self.scale_min_target = 0.005

        # Pruning
        self.prune_every = 2000
        self.prune_opacity_thresh = 0.01
        self.prune_min_gaussians = 2000

    def _build_params(self):
        """Build covariances, opacities from learnable parameters."""
        covariances = build_covariances(self.quaternions, self.log_scales)
        amplitudes = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0))
        opacities = amplitudes.clamp(0.0, 1.0)
        return covariances, opacities

    def train_step(self, camera_dict, gt_image):
        """
        One training step.

        Parameters
        ----------
        camera_dict : dict with 'R', 'T', 'fx', 'fy', 'cx', 'cy', 'width', 'height'
        gt_image : (H, W) ground truth
        """
        self.optimizer.zero_grad()

        H = camera_dict['height']
        W = camera_dict['width']

        # Build differentiable params
        covariances, opacities = self._build_params()

        # Apply aspect correction (differentiable)
        means_c, cov_c = apply_aspect_correction(
            self.means, covariances, self.aspect_scales
        )

        # Random pixel sampling
        N_pix = self.pixels_per_step
        total_pix = H * W
        if N_pix >= total_pix:
            ys = torch.arange(H, device=self.device, dtype=torch.float32) + 0.5
            xs = torch.arange(W, device=self.device, dtype=torch.float32) + 0.5
            gy, gx = torch.meshgrid(ys, xs, indexing='ij')
            pixels = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
            gt_vals = gt_image.reshape(-1)
        else:
            idx = torch.randperm(total_pix, device=self.device)[:N_pix]
            py = (idx // W).float() + 0.5
            px = (idx % W).float() + 0.5
            pixels = torch.stack([px, py], dim=-1)
            gt_vals = gt_image.reshape(-1)[idx]

        # colors = opacities for grayscale
        colors = opacities

        # CUDA splatting render
        rendered = self.renderer.render_at_pixels(
            means_c, cov_c, opacities, colors,
            camera_dict['R'], camera_dict['T'],
            camera_dict['fx'], camera_dict['fy'],
            camera_dict['cx'], camera_dict['cy'],
            W, H, pixels,
        )

        # L1 reconstruction loss
        l1 = F.l1_loss(rendered, gt_vals)

        # Opacity entropy regularization: push toward 0 or 1
        amp = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(1e-6, 1.0 - 1e-6)
        opacity_entropy = -(amp * torch.log(amp) + (1 - amp) * torch.log(1 - amp)).mean()
        opacity_reg = self.lambda_opacity * opacity_entropy

        # Scale regularization
        scales_cur = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_penalty = torch.clamp(self.scale_min_target - scales_cur, min=0.0).mean()
        scale_reg = self.lambda_scale * scale_penalty

        loss = l1 + opacity_reg + scale_reg
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            [self.means, self.log_scales, self.quaternions, self.log_amplitudes],
            max_norm=1.0,
        )

        self.optimizer.step()

        with torch.no_grad():
            self.log_scales.data.clamp_(-7.6, -1.2)
            self.log_amplitudes.data.clamp_(-9.2, 0.0)
            self.means.data.clamp_(-1.0, 1.0)

        return {
            'loss': loss.item(),
            'l1': l1.item(),
            'opacity_reg': opacity_reg.item(),
            'scale_reg': scale_reg.item(),
        }

    def prune_gaussians(self, step=0):
        """Remove low-opacity Gaussians."""
        with torch.no_grad():
            amp = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
            keep = amp > self.prune_opacity_thresh
            n_before = keep.shape[0]
            n_keep = keep.sum().item()

            if n_keep >= n_before or n_keep < self.prune_min_gaussians:
                return 0

            self.means = nn.Parameter(self.means.data[keep].clone())
            self.log_scales = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_amplitudes = nn.Parameter(self.log_amplitudes.data[keep].clone())

            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.Adam([
                {'params': [self.means], 'lr': lr},
                {'params': [self.log_scales], 'lr': lr * 0.5},
                {'params': [self.quaternions], 'lr': lr * 0.3},
                {'params': [self.log_amplitudes], 'lr': lr},
            ])

            n_pruned = n_before - n_keep
            print(f"  [Prune @ step {step}] {n_before} → {n_keep} "
                  f"(removed {n_pruned}, thresh={self.prune_opacity_thresh})")
            return n_pruned

    def save_checkpoint(self, path, step):
        """Save learnable parameters."""
        import os
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save({
            'means': self.means.data.cpu(),
            'log_scales': self.log_scales.data.cpu(),
            'quaternions': self.quaternions.data.cpu(),
            'log_amplitudes': self.log_amplitudes.data.cpu(),
            'step': step,
        }, path)

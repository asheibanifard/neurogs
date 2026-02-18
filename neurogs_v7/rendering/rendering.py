#!/usr/bin/env python3
"""
Gaussian Splatting Rendering Pipeline
======================================

Implements the 2D splatting formulation for rendering 3D Gaussians G(μ, Σ):

    1) Transform Gaussian to camera coordinate frame:
           μ_cam  ← R_cam · μ + T_cam
           Σ_cam  ← R_cam · Σ · R_camᵀ

    2) Project 3D Gaussian to 2D image plane:
           μ₂D  = prj(μ_cam) = (fx·μx/μz + cx,  fy·μy/μz + cy)

       Jacobian of the projection:
           J = [[fx/z,    0,  -fx·x/z²],
                [  0,   fy/z, -fy·y/z²]]

           Σ₂D = J · Σ_cam · Jᵀ

    3) 2D Gaussian evaluation:
           G₂D(u,v; μ₂D, Σ₂D) = exp(-½ (p-μ₂D)ᵀ Σ₂D⁻¹ (p-μ₂D))

    4) Splatting (image synthesis + optimisation):
           I(u,v) = Σᵢ wᵢ · G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾)

       Loss:
           min  Σₖ ||Iₖ(u,v) - Σᵢ wᵢ G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾)||²

Usage
-----
    from rendering.rendering import Camera, GaussianSplattingRenderer

    cam = Camera(fx=500, fy=500, cx=320, cy=240, width=640, height=480)
    renderer = GaussianSplattingRenderer()
    image = renderer.render(gaussians, cam, R, T)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field as dc_field
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================
#  Camera intrinsics
# ===================================================================
@dataclass
class Camera:
    """
    Pinhole camera model with intrinsic parameters.

    Attributes
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point (image centre) in pixels.
    width, height : int
        Image resolution.
    near, far : float
        Near/far clipping planes for depth culling.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    near: float = 0.1
    far: float = 100.0

    @property
    def K(self) -> torch.Tensor:
        """3×3 intrinsic matrix."""
        return torch.tensor([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,    1.0],
        ], dtype=torch.float32)

    @classmethod
    def from_fov(
        cls,
        fov_x_deg: float,
        width: int,
        height: int,
        near: float = 0.1,
        far: float = 100.0,
    ) -> "Camera":
        """Create camera from horizontal field-of-view (degrees)."""
        fx = width / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        fy = fx  # square pixels
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy,
                   width=width, height=height, near=near, far=far)


# ===================================================================
#  Gaussian parameters container
# ===================================================================
@dataclass
class GaussianParameters:
    """
    Container for 3D Gaussian primitive parameters.

    Attributes
    ----------
    means : (K, 3)
        3D centres μ_k in world coordinates.
    covariances : (K, 3, 3)
        3×3 covariance matrices Σ_k.
    weights : (K,)
        Per-Gaussian scalar weights/opacities w_k.
    colors : (K, C)
        Per-Gaussian colors (C channels, typically 3 for RGB or 1 for grayscale).
    """
    means: torch.Tensor          # (K, 3)
    covariances: torch.Tensor    # (K, 3, 3)
    weights: torch.Tensor        # (K,)
    colors: torch.Tensor         # (K, C)


# ===================================================================
#  Step 1: Transform Gaussian to camera coordinate frame
#
#     Given a 3D Gaussian G(μ, Σ) in world space:
#         μ_cam  ← R · μ + T
#         Σ_cam  ← R · Σ · Rᵀ
#
#     where R is the 3×3 world-to-camera rotation and T is translation.
# ===================================================================
def transform_to_camera(
    means: torch.Tensor,
    covariances: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform 3D Gaussians from world to camera coordinate frame.

        μ_cam ← R · μ + T
        Σ_cam ← R · Σ · Rᵀ

    Parameters
    ----------
    means : (K, 3)
        Gaussian centres in world coordinates.
    covariances : (K, 3, 3)
        3D covariance matrices in world frame.
    R : (3, 3)
        Camera rotation matrix (world → camera).
    T : (3,)
        Camera translation vector.

    Returns
    -------
    means_cam : (K, 3)
        Gaussian centres in camera coordinates.
    cov_cam : (K, 3, 3)
        Covariance matrices in camera frame.
    """
    # μ_cam = R @ μᵀ + T  →  (K, 3)
    means_cam = (R @ means.unsqueeze(-1)).squeeze(-1) + T.unsqueeze(0)

    # Σ_cam = R @ Σ @ Rᵀ  →  (K, 3, 3)
    # Broadcast: R is (3,3), covariances is (K,3,3)
    cov_cam = R.unsqueeze(0) @ covariances @ R.T.unsqueeze(0)

    return means_cam, cov_cam


# ===================================================================
#  Step 2: Project 3D Gaussian to 2D image plane
#
#     Projection of the camera-frame mean:
#         μ₂D = prj(μ_cam) = (u, v)
#             u = fx · μx/μz + cx
#             v = fy · μy/μz + cy
#
#     Jacobian of the projection mapping:
#         J = ∂(u,v)/∂(x,y,z) = ⎡ fx/z    0    -fx·x/z² ⎤
#                                ⎣   0   fy/z  -fy·y/z² ⎦
#
#     Projected 2D covariance (first-order approximation):
#         Σ₂D = J · Σ_cam · Jᵀ
# ===================================================================
def compute_projection_jacobian(
    means_cam: torch.Tensor,
    fx: float,
    fy: float,
) -> torch.Tensor:
    """
    Compute the Jacobian of pinhole projection at each Gaussian centre.

        J = ∂(u,v) / ∂(x,y,z) = [[fx/z,    0,  -fx·x/z²],
                                   [  0,   fy/z, -fy·y/z²]]

    Parameters
    ----------
    means_cam : (K, 3)
        Gaussian centres in camera frame [x, y, z].
    fx, fy : float
        Focal lengths.

    Returns
    -------
    J : (K, 2, 3)
        Per-Gaussian projection Jacobian.
    """
    x = means_cam[:, 0]  # (K,)
    y = means_cam[:, 1]  # (K,)
    z = means_cam[:, 2]  # (K,)

    z_sq = z * z
    # Clamp z to avoid division by zero for points at/behind camera
    z_safe = z.clamp(min=1e-6)
    z_sq_safe = z_sq.clamp(min=1e-12)

    K = means_cam.shape[0]
    J = torch.zeros(K, 2, 3, device=means_cam.device, dtype=means_cam.dtype)

    J[:, 0, 0] = fx / z_safe           # ∂u/∂x = fx / z
    J[:, 0, 1] = 0.0                   # ∂u/∂y = 0
    J[:, 0, 2] = -fx * x / z_sq_safe   # ∂u/∂z = -fx·x / z²

    J[:, 1, 0] = 0.0                   # ∂v/∂x = 0
    J[:, 1, 1] = fy / z_safe           # ∂v/∂y = fy / z
    J[:, 1, 2] = -fy * y / z_sq_safe   # ∂v/∂z = -fy·y / z²

    return J


def project_to_2d(
    means_cam: torch.Tensor,
    cov_cam: torch.Tensor,
    camera: Camera,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D Gaussians to 2D image plane.

        μ₂D = prj(μ_cam) = (fx·μx/μz + cx,  fy·μy/μz + cy)
        Σ₂D = J · Σ_cam · Jᵀ

    Parameters
    ----------
    means_cam : (K, 3)
        Gaussian centres in camera coordinates.
    cov_cam : (K, 3, 3)
        3D covariance in camera frame.
    camera : Camera
        Pinhole camera intrinsics.

    Returns
    -------
    means_2d : (K, 2)
        Projected 2D centres (u, v) in pixel coordinates.
    cov_2d : (K, 2, 2)
        Projected 2D covariance matrices Σ₂D.
    depths : (K,)
        Depth values (z-coordinate in camera frame) for sorting.
    """
    x = means_cam[:, 0]
    y = means_cam[:, 1]
    z = means_cam[:, 2]

    z_safe = z.clamp(min=1e-6)

    # μ₂D = (fx * x/z + cx,  fy * y/z + cy)
    u = camera.fx * x / z_safe + camera.cx
    v = camera.fy * y / z_safe + camera.cy
    means_2d = torch.stack([u, v], dim=-1)  # (K, 2)

    # Jacobian J: (K, 2, 3)
    J = compute_projection_jacobian(means_cam, camera.fx, camera.fy)

    # Σ₂D = J @ Σ_cam @ Jᵀ  →  (K, 2, 2)
    cov_2d = J @ cov_cam @ J.transpose(-2, -1)

    # Add small regularisation for numerical stability
    eps = 1e-4
    cov_2d = cov_2d + eps * torch.eye(2, device=cov_2d.device).unsqueeze(0)

    return means_2d, cov_2d, z


# ===================================================================
#  Step 3: Evaluate the 2D Gaussian
#
#     G₂D(u, v; μ₂D, Σ₂D) = exp( -½ (p - μ₂D)ᵀ · Σ₂D⁻¹ · (p - μ₂D) )
#
#     where p = (u, v) are pixel coordinates.
# ===================================================================
def evaluate_gaussian_2d(
    pixels: torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate 2D Gaussians at given pixel locations.

        G₂D(p; μ₂D, Σ₂D) = exp(-½ (p - μ₂D)ᵀ Σ₂D⁻¹ (p - μ₂D))

    Parameters
    ----------
    pixels : (N, 2)
        Pixel coordinates (u, v) to evaluate.
    means_2d : (K, 2)
        2D Gaussian centres.
    cov_2d : (K, 2, 2)
        2D covariance matrices.

    Returns
    -------
    values : (N, K)
        Gaussian values at each pixel for each primitive.
    """
    N = pixels.shape[0]
    K = means_2d.shape[0]

    # Difference: (N, K, 2) = pixels (N,1,2) - means (1,K,2)
    diff = pixels[:, None, :] - means_2d[None, :, :]  # (N, K, 2)

    # Invert 2×2 covariance: Σ⁻¹ for each Gaussian
    # For 2×2: [[a,b],[c,d]]⁻¹ = 1/det * [[d,-b],[-c,a]]
    a = cov_2d[:, 0, 0]  # (K,)
    b = cov_2d[:, 0, 1]
    c = cov_2d[:, 1, 0]
    d = cov_2d[:, 1, 1]

    det = a * d - b * c
    det_safe = det.clamp(min=1e-12)
    inv_det = 1.0 / det_safe

    # Σ⁻¹: (K, 2, 2) — built via torch.stack for safe gradient flow
    row0 = torch.stack([d * inv_det, -b * inv_det], dim=-1)   # (K, 2)
    row1 = torch.stack([-c * inv_det, a * inv_det], dim=-1)   # (K, 2)
    cov_inv = torch.stack([row0, row1], dim=-2)               # (K, 2, 2)

    # Mahalanobis distance: (p - μ)ᵀ Σ⁻¹ (p - μ)
    # diff: (N, K, 2), cov_inv: (1, K, 2, 2)
    # mahal = einsum('nki, kij, nkj -> nk', diff, cov_inv, diff)
    tmp = torch.einsum('nki,kij->nkj', diff, cov_inv)  # (N, K, 2)
    mahal = (tmp * diff).sum(dim=-1)                     # (N, K)

    values = torch.exp(-0.5 * mahal)  # (N, K)

    return values


def evaluate_gaussian_2d_batched(
    pixels: torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """
    Memory-efficient batched evaluation of 2D Gaussians.

    Chunks over pixels to avoid materialising the full (N, K) tensor.

    Parameters
    ----------
    pixels : (N, 2)
    means_2d : (K, 2)
    cov_2d : (K, 2, 2)
    chunk_size : int
        Number of pixels per chunk.

    Returns
    -------
    values : (N, K)
    """
    N = pixels.shape[0]
    chunks = []
    for i in range(0, N, chunk_size):
        chunk = evaluate_gaussian_2d(
            pixels[i:i + chunk_size], means_2d, cov_2d
        )
        chunks.append(chunk)
    return torch.cat(chunks, dim=0)


# ===================================================================
#  Step 4: Splatting — image synthesis & optimisation
#
#     Rendered image at pixel (u, v):
#         I(u,v) = Σᵢ wᵢ · G₂D⁽ⁱ⁾(u, v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾)
#
#     Optimisation objective:
#         min      Σₖ  ‖ Iₖ(u,v) − Σᵢ wᵢ G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾) ‖²
#       {μᵢ,Σᵢ,wᵢ}
#
#     where (u, v) depict the pixel coordinates.
# ===================================================================
def splat_gaussians(
    pixels: torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    weights: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
    sort_by_depth: bool = True,
) -> torch.Tensor:
    """
    Splat Gaussians onto pixel locations via weighted summation.

        I(u,v) = Σᵢ wᵢ · cᵢ · G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾)

    Parameters
    ----------
    pixels : (N, 2)
        Pixel coordinates to render.
    means_2d : (K, 2)
        Projected 2D centres.
    cov_2d : (K, 2, 2)
        Projected 2D covariances.
    weights : (K,)
        Per-Gaussian opacity/weight.
    colors : (K, C)
        Per-Gaussian color (C channels).
    depths : (K,)
        Depth for front-to-back ordering.
    sort_by_depth : bool
        If True, sort Gaussians front-to-back before compositing.

    Returns
    -------
    rendered : (N, C)
        Rendered pixel values.
    """
    K = means_2d.shape[0]

    # Sort by depth (front to back)
    if sort_by_depth:
        order = torch.argsort(depths)
        means_2d = means_2d[order]
        cov_2d = cov_2d[order]
        weights = weights[order]
        colors = colors[order]

    # Evaluate all 2D Gaussians at all pixels: (N, K)
    gauss_vals = evaluate_gaussian_2d(pixels, means_2d, cov_2d)

    # Weighted combination: I(u,v) = Σᵢ wᵢ · cᵢ · G₂D(...)
    # gauss_vals: (N, K), weights: (K,) → (1, K)
    # colors: (K, C) → (1, K, C)
    weighted = gauss_vals * weights[None, :]  # (N, K)

    # Render: Σ over K: (N, K) @ (K, C) → (N, C)
    rendered = weighted @ colors

    return rendered


def splat_gaussians_alpha(
    pixels: torch.Tensor,
    means_2d: torch.Tensor,
    cov_2d: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    depths: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Alpha-compositing splatting (front-to-back).

    Uses the standard volume rendering transmittance model:
        C(u,v) = Σᵢ Tᵢ · αᵢ · cᵢ
        Tᵢ = Π_{j<i} (1 - αⱼ)

    where αᵢ = oᵢ · G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾).

    Parameters
    ----------
    pixels : (N, 2)
    means_2d : (K, 2)
    cov_2d : (K, 2, 2)
    opacities : (K,)
        Per-Gaussian opacity in [0, 1].
    colors : (K, C)
    depths : (K,)

    Returns
    -------
    rendered : (N, C)
        Composited pixel colors.
    alpha_acc : (N,)
        Accumulated alpha (opacity) per pixel.
    """
    # Sort front-to-back by depth
    order = torch.argsort(depths)
    means_2d = means_2d[order]
    cov_2d = cov_2d[order]
    opacities = opacities[order]
    colors = colors[order]

    K = means_2d.shape[0]
    N = pixels.shape[0]
    C = colors.shape[1]

    # Evaluate 2D Gaussians: (N, K)
    gauss_vals = evaluate_gaussian_2d(pixels, means_2d, cov_2d)

    # Per-pixel per-Gaussian alpha: α_i(p) = o_i · G₂D(p)
    alpha = opacities[None, :] * gauss_vals  # (N, K)
    alpha = alpha.clamp(0.0, 0.999)

    # Front-to-back alpha compositing
    # T_i = prod_{j<i} (1 - alpha_j)
    one_minus_alpha = 1.0 - alpha  # (N, K)
    # Exclusive cumulative product along K dimension
    # T_i = cumprod of (1-alpha) shifted by 1 (T_0 = 1)
    T = torch.ones(N, K, device=pixels.device, dtype=pixels.dtype)
    T[:, 1:] = torch.cumprod(one_minus_alpha[:, :-1], dim=1)

    # Weight for each Gaussian at each pixel: T_i * alpha_i
    contribution = T * alpha  # (N, K)

    # Compose: C(p) = Σ_i T_i · α_i · c_i
    rendered = contribution @ colors  # (N, K) @ (K, C) → (N, C)

    # Accumulated alpha
    alpha_acc = 1.0 - torch.prod(one_minus_alpha, dim=1)  # (N,)

    return rendered, alpha_acc


# ===================================================================
#  Full rendering pipeline
# ===================================================================
class GaussianSplattingRenderer:
    """
    End-to-end differentiable Gaussian splatting renderer.

    Pipeline:
        3D Gaussians → camera transform → 2D projection → splatting → image
    """

    def __init__(
        self,
        tile_size: int = 16,
        radius_multiplier: float = 3.0,
        use_alpha_compositing: bool = True,
    ):
        """
        Parameters
        ----------
        tile_size : int
            Tile size for tiled rasterisation (not yet used; for future optimisation).
        radius_multiplier : float
            Multiplier on Gaussian radius for culling. A Gaussian is culled
            if its 2D footprint (radius_multiplier × sqrt(max eigenvalue))
            doesn't overlap the image.
        use_alpha_compositing : bool
            If True, use front-to-back alpha compositing.
            If False, use simple weighted summation.
        """
        self.tile_size = tile_size
        self.radius_multiplier = radius_multiplier
        self.use_alpha_compositing = use_alpha_compositing

    def _cull_gaussians(
        self,
        means_cam: torch.Tensor,
        means_2d: torch.Tensor,
        cov_2d: torch.Tensor,
        camera: Camera,
    ) -> torch.Tensor:
        """
        Visibility culling: remove Gaussians behind the camera or outside
        the image frustum.

        Returns a boolean mask of visible Gaussians.
        """
        K = means_cam.shape[0]
        device = means_cam.device

        # Depth culling: z must be in [near, far]
        z = means_cam[:, 2]
        depth_ok = (z > camera.near) & (z < camera.far)

        # Frustum culling: 2D centre + radius must overlap image
        # Approximate 2D radius from max eigenvalue of Σ₂D
        # For a 2×2 matrix, eigenvalues: λ = (tr ± sqrt(tr²-4det)) / 2
        a = cov_2d[:, 0, 0]
        b = cov_2d[:, 0, 1]
        d = cov_2d[:, 1, 1]
        tr = a + d
        det = a * d - b * b
        disc = (tr * tr - 4.0 * det).clamp(min=0.0)
        lambda_max = 0.5 * (tr + torch.sqrt(disc))
        radius = self.radius_multiplier * torch.sqrt(lambda_max.clamp(min=1e-8))

        u = means_2d[:, 0]
        v = means_2d[:, 1]
        in_frustum = (
            (u + radius > 0) & (u - radius < camera.width) &
            (v + radius > 0) & (v - radius < camera.height)
        )

        return depth_ok & in_frustum

    def render(
        self,
        gaussians: GaussianParameters,
        camera: Camera,
        R: torch.Tensor,
        T: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Render a full image from 3D Gaussians.

        Parameters
        ----------
        gaussians : GaussianParameters
            3D Gaussian primitives.
        camera : Camera
            Pinhole camera intrinsics.
        R : (3, 3)
            World-to-camera rotation.
        T : (3,)
            World-to-camera translation.
        background : (C,) or None
            Background color. Defaults to zeros.

        Returns
        -------
        dict with keys:
            'image' : (H, W, C) rendered image
            'alpha' : (H, W) accumulated alpha (if alpha compositing)
            'depth' : (H, W) expected depth (if alpha compositing)
            'n_visible' : int, number of visible Gaussians
        """
        device = gaussians.means.device
        C = gaussians.colors.shape[1]
        H, W = camera.height, camera.width

        if background is None:
            background = torch.zeros(C, device=device)

        # --- Step 1: World → Camera ---
        means_cam, cov_cam = transform_to_camera(
            gaussians.means, gaussians.covariances, R, T,
        )

        # --- Step 2: 3D → 2D projection ---
        means_2d, cov_2d, depths = project_to_2d(means_cam, cov_cam, camera)

        # --- Visibility culling ---
        with torch.no_grad():
            visible = self._cull_gaussians(means_cam, means_2d, cov_2d, camera)
        n_visible = int(visible.sum().item())

        if n_visible == 0:
            image = background.unsqueeze(0).unsqueeze(0).expand(H, W, C)
            return {
                'image': image,
                'alpha': torch.zeros(H, W, device=device),
                'n_visible': 0,
            }

        # Filter to visible Gaussians
        means_2d_vis = means_2d[visible]
        cov_2d_vis = cov_2d[visible]
        weights_vis = gaussians.weights[visible]
        colors_vis = gaussians.colors[visible]
        depths_vis = depths[visible]

        # --- Build pixel grid ---
        # Create (H*W, 2) pixel coordinate tensor
        ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        # --- Step 3 & 4: Splatting ---
        if self.use_alpha_compositing:
            rendered, alpha_acc = splat_gaussians_alpha(
                pixels, means_2d_vis, cov_2d_vis,
                weights_vis, colors_vis, depths_vis,
            )
            # Reshape to image
            rendered = rendered.reshape(H, W, C)
            alpha_acc = alpha_acc.reshape(H, W)

            # Composite with background
            image = rendered + (1.0 - alpha_acc.unsqueeze(-1)) * background

            return {
                'image': image,
                'alpha': alpha_acc,
                'n_visible': n_visible,
            }
        else:
            rendered = splat_gaussians(
                pixels, means_2d_vis, cov_2d_vis,
                weights_vis, colors_vis, depths_vis,
            )
            image = rendered.reshape(H, W, C)

            return {
                'image': image,
                'n_visible': n_visible,
            }

    def render_at_pixels(
        self,
        gaussians: GaussianParameters,
        camera: Camera,
        R: torch.Tensor,
        T: torch.Tensor,
        pixels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Render at specific pixel locations (for training with random sampling).

        Parameters
        ----------
        gaussians : GaussianParameters
        camera : Camera
        R : (3, 3) rotation
        T : (3,) translation
        pixels : (N, 2) pixel coordinates

        Returns
        -------
        rendered : (N, C) rendered values at the given pixels.
        """
        # Step 1
        means_cam, cov_cam = transform_to_camera(
            gaussians.means, gaussians.covariances, R, T,
        )

        # Step 2
        means_2d, cov_2d, depths = project_to_2d(means_cam, cov_cam, camera)

        # Cull (detached — mask only, no grad needed)
        with torch.no_grad():
            visible = self._cull_gaussians(means_cam, means_2d, cov_2d, camera)
        if visible.sum() == 0:
            # Return differentiable zero so .backward() doesn't crash
            return gaussians.weights.sum() * 0.0 + torch.zeros(
                pixels.shape[0], gaussians.colors.shape[1],
                device=pixels.device)

        means_2d_vis = means_2d[visible]
        cov_2d_vis = cov_2d[visible]
        weights_vis = gaussians.weights[visible]
        colors_vis = gaussians.colors[visible]
        depths_vis = depths[visible]

        # Steps 3 & 4
        if self.use_alpha_compositing:
            rendered, _ = splat_gaussians_alpha(
                pixels, means_2d_vis, cov_2d_vis,
                weights_vis, colors_vis, depths_vis,
            )
        else:
            rendered = splat_gaussians(
                pixels, means_2d_vis, cov_2d_vis,
                weights_vis, colors_vis, depths_vis,
            )

        return rendered


# ===================================================================
#  Splatting loss (Step 4 optimisation objective)
# ===================================================================
def splatting_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Compute the splatting reconstruction loss.

        L = Σₖ ||Iₖ(u,v) - Σᵢ wᵢ G₂D(u,v; μ₂D⁽ⁱ⁾, Σ₂D⁽ⁱ⁾)||²

    Parameters
    ----------
    rendered : (N, C) or (H, W, C)
        Rendered pixel values.
    target : same shape as rendered
        Ground-truth pixel values.
    loss_type : str
        'l2' for MSE, 'l1' for L1, 'charbonnier' for Charbonnier.

    Returns
    -------
    loss : scalar tensor
    """
    if loss_type == "l2":
        return F.mse_loss(rendered, target)
    elif loss_type == "l1":
        return F.l1_loss(rendered, target)
    elif loss_type == "charbonnier":
        eps = 1e-6
        diff = rendered - target
        return torch.sqrt(diff * diff + eps * eps).mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def ssim_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    """
    Compute 1 - SSIM loss for image-level supervision.

    Parameters
    ----------
    rendered : (H, W, C) rendered image.
    target : (H, W, C) ground-truth image.

    Returns
    -------
    loss : scalar, 1 - SSIM.
    """
    # Reshape to (1, C, H, W) for conv2d
    C = rendered.shape[-1]
    pred = rendered.permute(2, 0, 1).unsqueeze(0)
    gt = target.permute(2, 0, 1).unsqueeze(0)

    # Gaussian window
    coords = torch.arange(window_size, device=rendered.device, dtype=torch.float32)
    coords -= window_size // 2
    g = torch.exp(-coords ** 2 / (2.0 * 1.5 ** 2))
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window / window.sum()
    window = window.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)

    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
    mu_gt = F.conv2d(gt, window, padding=pad, groups=C)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq = mu_gt ** 2
    mu_cross = mu_pred * mu_gt

    sigma_pred = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
    sigma_gt = F.conv2d(gt ** 2, window, padding=pad, groups=C) - mu_gt_sq
    sigma_cross = F.conv2d(pred * gt, window, padding=pad, groups=C) - mu_cross

    ssim_map = (
        (2.0 * mu_cross + C1) * (2.0 * sigma_cross + C2)
    ) / (
        (mu_pred_sq + mu_gt_sq + C1) * (sigma_pred + sigma_gt + C2)
    )

    return 1.0 - ssim_map.mean()


# ===================================================================
#  Helper: extract GaussianParameters from GaussianMixtureField model
# ===================================================================
def gaussians_from_model(
    model: nn.Module,
    colors: Optional[torch.Tensor] = None,
) -> GaussianParameters:
    """
    Extract GaussianParameters from a GaussianMixtureField model.

    Parameters
    ----------
    model : nn.Module
        Must have attributes: means, log_scales, quaternions, log_amplitudes,
        and methods: quat_to_rotmat(), get_covariance_matrices().
    colors : (K, C) or None
        Per-Gaussian colors. If None, uses amplitudes as grayscale.

    Returns
    -------
    GaussianParameters
    """
    with torch.no_grad():
        means = model.means.data
        covariances = model.get_covariance_matrices().data
        weights = torch.exp(model.log_amplitudes.data).clamp(0.0, 1.0)

        if colors is None:
            # Grayscale: use amplitude as intensity
            colors = weights.unsqueeze(-1)  # (K, 1)

    return GaussianParameters(
        means=means,
        covariances=covariances,
        weights=weights,
        colors=colors,
    )


# ===================================================================
#  Convenience: orbit camera poses
# ===================================================================
def orbit_camera_pose(
    elevation_deg: float,
    azimuth_deg: float,
    radius: float,
    target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute camera extrinsics (R, T) for an orbit camera.

    Parameters
    ----------
    elevation_deg : float
        Camera elevation angle in degrees (0 = horizon, 90 = top-down).
    azimuth_deg : float
        Camera azimuth angle in degrees.
    radius : float
        Distance from the target/origin.
    target : (3,) or None
        Look-at target point. Defaults to origin.

    Returns
    -------
    R : (3, 3) rotation matrix (world → camera).
    T : (3,) translation vector.
    """
    if target is None:
        target = torch.zeros(3)

    el = math.radians(elevation_deg)
    az = math.radians(azimuth_deg)

    # Camera position in world coordinates
    cam_x = radius * math.cos(el) * math.sin(az)
    cam_y = radius * math.sin(el)
    cam_z = radius * math.cos(el) * math.cos(az)
    cam_pos = torch.tensor([cam_x, cam_y, cam_z], dtype=torch.float32)

    # Forward vector: camera → target
    forward = target - cam_pos
    forward = forward / (forward.norm() + 1e-8)

    # Up vector (world Y-up)
    world_up = torch.tensor([0.0, 1.0, 0.0])

    # Right = forward × up
    right = torch.linalg.cross(forward, world_up)
    if right.norm() < 1e-6:
        # Camera looking straight down or up
        world_up = torch.tensor([0.0, 0.0, 1.0])
        right = torch.linalg.cross(forward, world_up)
    right = right / (right.norm() + 1e-8)

    # Recompute up = right × forward
    up = torch.linalg.cross(right, forward)
    up = up / (up.norm() + 1e-8)

    # Build rotation matrix — OpenCV convention:
    #   x_cam = right,  y_cam = -up (down),  z_cam = forward (into scene)
    # This ensures points in front of the camera have positive z,
    # consistent with the projection μ₂D = (fx·x/z + cx, fy·y/z + cy).
    R = torch.stack([right, -up, forward], dim=0)  # (3, 3)

    # Translation: T = -R @ cam_pos
    T = -R @ cam_pos

    return R, T


# ===================================================================
#  Aspect-ratio correction for anisotropic volumes
# ===================================================================
def compute_aspect_scales(
    vol_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """
    Compute per-axis scale factors that preserve the original volume's
    aspect ratio when going from the isotropic [-1,1]³ training space
    to a proportional world coordinate system.

    The largest dimension maps to [-1, 1]; shorter dimensions map to a
    proportionally smaller range.

    Parameters
    ----------
    vol_shape : (Z, Y, X)
        Shape of the original 3D volume.

    Returns
    -------
    aspect_scales : (3,) tensor  [sx, sy, sz]
        Multiply the normalised means (in [-1,1]³) by these factors to
        recover the original aspect ratio.
        Convention: index 0 = X, 1 = Y, 2 = Z  (matches Gaussian mean order).
    """
    Z, Y, X = vol_shape
    max_dim = float(max(X, Y, Z))
    return torch.tensor([X / max_dim, Y / max_dim, Z / max_dim],
                        dtype=torch.float32)


def apply_aspect_correction(
    gaussians: "GaussianParameters",
    aspect_scales: torch.Tensor,
) -> "GaussianParameters":
    """
    Transform GaussianParameters from isotropic [-1,1]³ space to
    aspect-corrected world space.

    Parameters
    ----------
    gaussians : GaussianParameters
        In the uniform [-1,1]³ coordinate system.
    aspect_scales : (3,)
        From `compute_aspect_scales()`.

    Returns
    -------
    GaussianParameters with corrected means and covariances.
    """
    s = aspect_scales.to(gaussians.means.device)  # (3,)
    means_corrected = gaussians.means * s.unsqueeze(0)  # (K, 3)

    # Σ_corrected = S Σ Sᵀ  where S = diag(s)
    S = torch.diag(s)  # (3, 3)
    cov_corrected = S.unsqueeze(0) @ gaussians.covariances @ S.unsqueeze(0).transpose(-2, -1)

    return GaussianParameters(
        means=means_corrected,
        covariances=cov_corrected,
        weights=gaussians.weights,
        colors=gaussians.colors,
    )


# ===================================================================
#  Multi-view MIP: generate GT projections from 3D volume
# ===================================================================
def load_volume(tif_path: str) -> np.ndarray:
    """Load a 3D TIFF volume, normalise to [0, 1] float32."""
    import tifffile
    vol = tifffile.imread(tif_path).astype(np.float32)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    return ((vol - vmin) / (vmax - vmin)).astype(np.float32)


def _sample_volume_trilinear(
    vol: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """
    Trilinear-interpolated sampling of a 3D volume.

    Parameters
    ----------
    vol : (1, 1, Z, Y, X)  or  (Z, Y, X)
        Normalised volume on GPU.
    points : (N, 3)
        Query points in normalised [-1, 1]³ world space.
        Convention: points[:, 0]=x, [:, 1]=y, [:, 2]=z.

    Returns
    -------
    values : (N,)
        Interpolated volume intensities.
    """
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)  # (1, 1, Z, Y, X)

    N = points.shape[0]
    # grid_sample expects (N, D_out, H_out, W_out, 3) with order (x, y, z)
    # mapping: x→W(X), y→H(Y), z→D(Z)  — already matches our convention
    grid = points.reshape(1, 1, 1, N, 3)  # (1, 1, 1, N, 3)
    sampled = F.grid_sample(
        vol, grid, mode='bilinear', padding_mode='zeros', align_corners=True,
    )
    return sampled.reshape(N)


def render_mip_from_camera(
    vol: torch.Tensor,
    camera: "Camera",
    R: torch.Tensor,
    T: torch.Tensor,
    n_samples: int = 256,
    near: float = 0.1,
    far: float = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Render a Maximum Intensity Projection from the volume along camera rays.

    For each pixel, cast a ray through the volume, sample at `n_samples`
    points along the ray, and take the max intensity.

    Parameters
    ----------
    vol : (Z, Y, X) or (1, 1, Z, Y, X)
        Normalised volume on GPU.
    camera : Camera
        Pinhole camera intrinsics.
    R : (3, 3)
        World-to-camera rotation.
    T : (3,)
        World-to-camera translation.
    n_samples : int
        Number of samples along each ray.
    near, far : float
        Ray marching range (in camera space z).
    aspect_scales : (3,) tensor or None
        If provided, ray sample points are in aspect-corrected world space.
        They are divided by aspect_scales before querying the volume via
        grid_sample (which expects isotropic [-1,1]³).

    Returns
    -------
    mip_image : (H, W)
        MIP projection image (grayscale, [0, 1]).
    """
    device = vol.device if isinstance(vol, torch.Tensor) else torch.device('cpu')
    if not isinstance(vol, torch.Tensor):
        vol = torch.from_numpy(vol).to(device)
    if vol.ndim == 3:
        vol = vol.unsqueeze(0).unsqueeze(0)

    H, W = camera.height, camera.width

    # Camera-to-world: R_cw = R^T, T_cw = -R^T @ T
    R_cw = R.T  # (3, 3)
    T_cw = -R_cw @ T  # (3,)  = camera position in world

    # Build ray directions for each pixel (in camera space)
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    # Pixel → camera-space direction
    dir_cam_x = (grid_x - camera.cx) / camera.fx
    dir_cam_y = (grid_y - camera.cy) / camera.fy
    dir_cam_z = torch.ones_like(dir_cam_x)
    dirs_cam = torch.stack([dir_cam_x, dir_cam_y, dir_cam_z], dim=-1)  # (H, W, 3)

    # Camera-space direction → world-space direction
    dirs_world = (R_cw @ dirs_cam.reshape(-1, 3).T).T  # (H*W, 3)
    dirs_world = dirs_world / (dirs_world.norm(dim=-1, keepdim=True) + 1e-8)

    # Ray origin (camera position in world)
    origin = T_cw.unsqueeze(0)  # (1, 3)

    # Sample t-values along each ray
    t_vals = torch.linspace(near, far, n_samples, device=device)  # (S,)

    # Build all sample points: (H*W, S, 3)
    N_rays = H * W
    # points = origin + t * dir  →  (N_rays, S, 3)
    points = origin.unsqueeze(1) + t_vals[None, :, None] * dirs_world.unsqueeze(1)

    # Sample volume in chunks to save memory
    chunk_size = 1024  # rays per chunk
    mip_flat = torch.zeros(N_rays, device=device)

    for i in range(0, N_rays, chunk_size):
        j = min(i + chunk_size, N_rays)
        pts_chunk = points[i:j]  # (chunk, S, 3)
        chunk_rays = j - i

        pts_flat = pts_chunk.reshape(-1, 3)  # (chunk*S, 3)
        # If aspect-corrected, convert back to isotropic [-1,1]³ for grid_sample
        if aspect_scales is not None:
            inv_scales = (1.0 / aspect_scales.to(pts_flat.device)).unsqueeze(0)
            pts_flat = pts_flat * inv_scales
        vals = _sample_volume_trilinear(vol, pts_flat)  # (chunk*S,)
        vals = vals.reshape(chunk_rays, n_samples)       # (chunk, S)
        mip_flat[i:j] = vals.max(dim=1)[0]              # MIP: take max

    return mip_flat.reshape(H, W)


def generate_camera_poses(
    n_azimuth: int = 12,
    n_elevation: int = 5,
    elevation_range: Tuple[float, float] = (-60.0, 60.0),
    radius: float = 3.5,
    include_axis_aligned: bool = True,
) -> List[dict]:
    """
    Generate a set of orbit camera poses covering many viewpoints.

    Parameters
    ----------
    n_azimuth : int
        Number of azimuth angles per elevation ring.
    n_elevation : int
        Number of elevation levels.
    elevation_range : (float, float)
        Min/max elevation in degrees.
    radius : float
        Camera distance from origin.
    include_axis_aligned : bool
        If True, also add the 6 axis-aligned views (±X, ±Y, ±Z).

    Returns
    -------
    poses : list of dict
        Each dict has keys: 'R' (3,3), 'T' (3,), 'elevation', 'azimuth'.
    """
    poses = []

    # Orbit grid
    elevations = np.linspace(
        elevation_range[0], elevation_range[1], n_elevation
    )
    azimuths = np.linspace(0, 360, n_azimuth, endpoint=False)

    for el in elevations:
        for az in azimuths:
            R, T = orbit_camera_pose(float(el), float(az), radius)
            poses.append({
                'R': R, 'T': T,
                'elevation': float(el),
                'azimuth': float(az),
            })

    # Axis-aligned views: ±X, ±Y, ±Z
    if include_axis_aligned:
        axis_views = [
            (0.0, 0.0),    # +Z  (front)
            (0.0, 180.0),  # -Z  (back)
            (0.0, 90.0),   # +X  (right)
            (0.0, -90.0),  # -X  (left)
            (89.0, 0.0),   # +Y  (top)   — use 89° to avoid gimbal lock
            (-89.0, 0.0),  # -Y  (bottom)
        ]
        for el, az in axis_views:
            R, T = orbit_camera_pose(el, az, radius)
            poses.append({
                'R': R, 'T': T,
                'elevation': el,
                'azimuth': az,
            })

    return poses


def generate_mip_dataset(
    vol: torch.Tensor,
    camera: "Camera",
    poses: List[dict],
    n_ray_samples: int = 256,
    near: float = 0.5,
    far: float = 6.0,
    aspect_scales: Optional[torch.Tensor] = None,
) -> List[dict]:
    """
    Generate a full multi-view MIP dataset from the 3D volume.

    Parameters
    ----------
    vol : (Z, Y, X)
        Normalised 3D volume on GPU.
    camera : Camera
        Shared pinhole camera intrinsics.
    poses : list of dict
        Camera poses from `generate_camera_poses()`.
    n_ray_samples : int
        Samples per ray for MIP rendering.
    near, far : float
        Ray marching bounds.
    aspect_scales : (3,) tensor or None
        Aspect-ratio correction factors (from `compute_aspect_scales()`).
        Passed through to `render_mip_from_camera()`.

    Returns
    -------
    dataset : list of dict
        Each entry has:
            'image'     : (H, W) GT MIP tensor
            'R'         : (3, 3) rotation
            'T'         : (3,) translation
            'elevation' : float
            'azimuth'   : float
    """
    dataset = []
    device = vol.device

    for idx, pose in enumerate(poses):
        R = pose['R'].to(device)
        T = pose['T'].to(device)

        mip_img = render_mip_from_camera(
            vol, camera, R, T,
            n_samples=n_ray_samples, near=near, far=far,
            aspect_scales=aspect_scales,
        )

        dataset.append({
            'image': mip_img,        # (H, W)
            'R': R,
            'T': T,
            'elevation': pose['elevation'],
            'azimuth': pose['azimuth'],
        })

        if (idx + 1) % 10 == 0 or idx == len(poses) - 1:
            print(f"  MIP dataset: {idx+1}/{len(poses)} views rendered")

    return dataset


# ===================================================================
#  Splatting training loop with MIP ground truth
# ===================================================================
class SplattingTrainer:
    """
    Train 3D Gaussians against multi-view MIP ground truth.

    Optimises (means, log_scales, quaternions, log_amplitudes) so that
    the splatted 2D projections match the MIP images from the volume.
    """

    def __init__(
        self,
        means: torch.Tensor,
        log_scales: torch.Tensor,
        quaternions: torch.Tensor,
        log_amplitudes: torch.Tensor,
        lr: float = 1e-3,
        lambda_ssim: float = 0.2,
        pixels_per_step: int = 8192,
        aspect_scales: Optional[torch.Tensor] = None,
    ):
        """
        Parameters
        ----------
        means, log_scales, quaternions, log_amplitudes :
            Learnable Gaussian parameters (will be wrapped in nn.Parameter).
        lr : float
            Learning rate.
        lambda_ssim : float
            Weight for SSIM loss component (0 = pure L1).
        pixels_per_step : int
            Number of randomly sampled pixels per training step.
        aspect_scales : (3,) tensor or None
            If provided (from `compute_aspect_scales()`), the Gaussian means
            and covariances are transformed to aspect-corrected world space
            before rendering, so the splatted image matches the MIP GT that
            was produced in the same corrected space.
        """
        self.device = means.device

        self.means = nn.Parameter(means.clone())
        self.log_scales = nn.Parameter(log_scales.clone())
        self.quaternions = nn.Parameter(quaternions.clone())
        self.log_amplitudes = nn.Parameter(log_amplitudes.clone())

        self.optimizer = torch.optim.Adam([
            {'params': [self.means], 'lr': lr},
            {'params': [self.log_scales], 'lr': lr * 0.5},
            {'params': [self.quaternions], 'lr': lr * 0.3},
            {'params': [self.log_amplitudes], 'lr': lr},
        ])

        self.lambda_ssim = lambda_ssim
        self.pixels_per_step = pixels_per_step
        self.aspect_scales = aspect_scales  # (3,) or None
        self.renderer = GaussianSplattingRenderer(use_alpha_compositing=True)

        # Regularization hyperparameters
        self.lambda_opacity = 0.01   # opacity entropy: push toward 0 or 1
        self.lambda_scale = 0.001    # penalise very small Gaussians
        self.scale_min_target = 0.005 # scale below this is penalised

        # Pruning settings
        self.prune_every = 1000      # prune every N steps
        self.prune_opacity_thresh = 0.005  # remove Gaussians dimmer than this
        self.prune_min_gaussians = 2000    # never prune below this count

    def _build_gaussians(self) -> GaussianParameters:
        """Reconstruct GaussianParameters from current learnable params."""
        K = self.means.shape[0]
        scales = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        q = F.normalize(self.quaternions, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros(K, 3, 3, device=self.device, dtype=q.dtype)
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
        covariances = R @ S2 @ R.transpose(-2, -1)

        amplitudes = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0))
        weights = amplitudes.clamp(0.0, 1.0)
        colors = weights.unsqueeze(-1)  # (K, 1) grayscale

        return GaussianParameters(
            means=self.means,
            covariances=covariances,
            weights=weights,
            colors=colors,
        )

    def _build_gaussians_corrected(self) -> GaussianParameters:
        """Build GaussianParameters with aspect-ratio correction applied."""
        g = self._build_gaussians()
        if self.aspect_scales is not None:
            g = apply_aspect_correction(g, self.aspect_scales)
        return g

    def train_step(
        self,
        camera: "Camera",
        gt_image: torch.Tensor,
        R_cam: torch.Tensor,
        T_cam: torch.Tensor,
    ) -> dict:
        """
        One training step: render splatted image, compare to GT MIP.

        Parameters
        ----------
        camera : Camera
        gt_image : (H, W) ground-truth MIP image.
        R_cam, T_cam : camera extrinsics.

        Returns
        -------
        dict with 'loss', 'l1', 'ssim', 'n_visible'.
        """
        self.optimizer.zero_grad()

        gaussians = self._build_gaussians_corrected()
        H, W = camera.height, camera.width

        # Random pixel sampling for efficiency
        N_pix = self.pixels_per_step
        total_pix = H * W
        if N_pix >= total_pix:
            # Use all pixels
            ys = torch.arange(H, device=self.device, dtype=torch.float32) + 0.5
            xs = torch.arange(W, device=self.device, dtype=torch.float32) + 0.5
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
            pixels = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
            gt_vals = gt_image.reshape(-1)
        else:
            # Random subset of pixels
            idx = torch.randperm(total_pix, device=self.device)[:N_pix]
            py = (idx // W).float() + 0.5
            px = (idx % W).float() + 0.5
            pixels = torch.stack([px, py], dim=-1)  # (N, 2)
            gt_vals = gt_image.reshape(-1)[idx]      # (N,)

        # Render at sampled pixels
        rendered = self.renderer.render_at_pixels(
            gaussians, camera, R_cam, T_cam, pixels,
        )  # (N, 1) grayscale
        pred_vals = rendered.squeeze(-1)  # (N,)

        # L1 loss
        l1 = F.l1_loss(pred_vals, gt_vals)

        # Opacity regularisation: binary cross-entropy pushes toward 0 or 1
        # H(α) = -α log(α) - (1-α) log(1-α), maximised at α=0.5
        amp = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(1e-6, 1.0 - 1e-6)
        opacity_entropy = -(amp * torch.log(amp) + (1 - amp) * torch.log(1 - amp)).mean()
        opacity_reg = self.lambda_opacity * opacity_entropy

        # Scale regularisation: penalise scales smaller than threshold
        scales_cur = torch.exp(self.log_scales).clamp(1e-5, 1e2)
        scale_penalty = torch.clamp(self.scale_min_target - scales_cur, min=0.0).mean()
        scale_reg = self.lambda_scale * scale_penalty

        # Total loss
        loss = l1 + opacity_reg + scale_reg

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [self.means, self.log_scales, self.quaternions, self.log_amplitudes],
            max_norm=1.0,
        )

        self.optimizer.step()

        # Clamp parameters
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

    def prune_gaussians(self, step: int = 0):
        """
        Remove Gaussians with opacity below threshold.
        Rebuilds all parameter tensors and the optimiser.
        """
        with torch.no_grad():
            amp = torch.exp(self.log_amplitudes.clamp(-10.0, 6.0)).clamp(0.0, 1.0)
            keep = amp > self.prune_opacity_thresh
            n_before = keep.shape[0]
            n_keep = keep.sum().item()

            if n_keep >= n_before:
                return 0  # nothing to prune
            if n_keep < self.prune_min_gaussians:
                # Don't prune below minimum
                return 0

            self.means = nn.Parameter(self.means.data[keep].clone())
            self.log_scales = nn.Parameter(self.log_scales.data[keep].clone())
            self.quaternions = nn.Parameter(self.quaternions.data[keep].clone())
            self.log_amplitudes = nn.Parameter(self.log_amplitudes.data[keep].clone())

            # Rebuild optimiser with new parameters
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = torch.optim.Adam([
                {'params': [self.means], 'lr': lr},
                {'params': [self.log_scales], 'lr': lr * 0.5},
                {'params': [self.quaternions], 'lr': lr * 0.3},
                {'params': [self.log_amplitudes], 'lr': lr},
            ])

            n_pruned = n_before - n_keep
            print(f"  [Prune @ step {step}] Removed {n_pruned} Gaussians "
                  f"(amp < {self.prune_opacity_thresh}): {n_before} → {n_keep}")
            return n_pruned

    def train(
        self,
        camera: "Camera",
        dataset: List[dict],
        n_steps: int = 10000,
        log_every: int = 50,
        save_path: Optional[str] = None,
        save_every: int = 2000,
    ) -> List[dict]:
        """
        Full training loop over multi-view MIP dataset.

        Parameters
        ----------
        camera : Camera
        dataset : list of dict from `generate_mip_dataset()`.
        n_steps : int
            Total training iterations.
        log_every : int
            Print loss every N steps.
        save_path : str or None
            Checkpoint save path template (e.g. 'ckpt/splat_{step}.pt').
        save_every : int
            Save checkpoint every N steps.

        Returns
        -------
        history : list of dict with per-step metrics.
        """
        n_views = len(dataset)
        history = []

        print(f"\nSplatting training: {n_steps} steps, {n_views} views, "
              f"{self.pixels_per_step} pixels/step")
        print(f"  Gaussians: {self.means.shape[0]}")
        print("-" * 60)

        for step in range(1, n_steps + 1):
            # Periodic pruning
            if self.prune_every > 0 and step % self.prune_every == 0 and step > 0:
                self.prune_gaussians(step)

            # Random view selection
            view_idx = torch.randint(0, n_views, (1,)).item()
            view = dataset[view_idx]

            metrics = self.train_step(
                camera,
                view['image'],
                view['R'],
                view['T'],
            )
            history.append(metrics)

            if step % log_every == 0:
                avg_loss = np.mean([h['loss'] for h in history[-log_every:]])
                avg_l1 = np.mean([h['l1'] for h in history[-log_every:]])
                n_gauss = self.means.shape[0]
                print(f"  Step {step:>6d}/{n_steps}  |  loss={avg_loss:.6f}  "
                      f"l1={avg_l1:.6f}  K={n_gauss}")

            if save_path and step % save_every == 0:
                ckpt = {
                    'means': self.means.data.cpu(),
                    'log_scales': self.log_scales.data.cpu(),
                    'quaternions': self.quaternions.data.cpu(),
                    'log_amplitudes': self.log_amplitudes.data.cpu(),
                    'step': step,
                }
                path = save_path.format(step=step)
                os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
                torch.save(ckpt, path)
                print(f"  Checkpoint saved → {path}")

        # Final save
        if save_path:
            ckpt = {
                'means': self.means.data.cpu(),
                'log_scales': self.log_scales.data.cpu(),
                'quaternions': self.quaternions.data.cpu(),
                'log_amplitudes': self.log_amplitudes.data.cpu(),
                'step': n_steps,
            }
            path = save_path.format(step=n_steps)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(ckpt, path)
            print(f"  Final checkpoint → {path}")

        return history


# ===================================================================
#  Main: generate MIP dataset & train splatting from checkpoint
# ===================================================================
if __name__ == "__main__":
    print("Gaussian Splatting — MIP-supervised Training Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    #  1. Load 3D volume (ground truth for MIP generation)
    # ------------------------------------------------------------------
    vol_path = os.path.join(os.path.dirname(__file__), "..",
                            "10-2900-control-cell-05_cropped_corrected.tif")
    vol_path = os.path.abspath(vol_path)
    print(f"Loading volume: {vol_path}")
    vol_np = load_volume(vol_path)
    Z, Y, X = vol_np.shape
    print(f"  Volume shape (Z,Y,X): ({Z}, {Y}, {X})")
    vol_gpu = torch.from_numpy(vol_np).to(device)

    # ------------------------------------------------------------------
    #  1b. Compute aspect-ratio correction
    #      Volume (Z=100, Y=647, X=813) is far from cubic — we need to
    #      preserve the original proportions in world space.
    # ------------------------------------------------------------------
    aspect_scales = compute_aspect_scales((Z, Y, X))
    print(f"  Aspect scales (x,y,z): {aspect_scales.tolist()}")

    # ------------------------------------------------------------------
    #  2. Load trained Gaussian parameters from checkpoint
    # ------------------------------------------------------------------
    ckpt_path = os.path.join(os.path.dirname(__file__), "..",
                             "checkpoints", "gmf_refined_best.pt")
    ckpt_path = os.path.abspath(ckpt_path)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    means = ckpt["means"]
    log_scales = ckpt["log_scales"]
    quaternions = ckpt["quaternions"]
    log_amplitudes = ckpt["log_amplitudes"]
    K = means.shape[0]
    print(f"  {K} Gaussians loaded")

    # ------------------------------------------------------------------
    #  3. Set up camera (shared for all views)
    #     Scene lives in [-1, 1]³ → radius=3.5 ensures full coverage
    # ------------------------------------------------------------------
    H, W = 256, 256
    camera = Camera.from_fov(fov_x_deg=50.0, width=W, height=H,
                             near=0.01, far=10.0)
    print(f"  Camera: {W}×{H}, fx={camera.fx:.1f}, fy={camera.fy:.1f}")

    # ------------------------------------------------------------------
    #  4. Generate multi-view camera poses
    #     12 azimuths × 5 elevations = 60 orbit views + 6 axis-aligned = 66
    # ------------------------------------------------------------------
    print("\nGenerating camera poses...")
    poses = generate_camera_poses(
        n_azimuth=12,
        n_elevation=5,
        elevation_range=(-60.0, 60.0),
        radius=3.5,
        include_axis_aligned=True,
    )
    print(f"  {len(poses)} camera poses generated")

    # ------------------------------------------------------------------
    #  5. Render MIP ground truth from every viewpoint
    # ------------------------------------------------------------------
    print("\nRendering MIP ground-truth dataset...")
    dataset = generate_mip_dataset(
        vol_gpu, camera, poses,
        n_ray_samples=200,   # dense sampling for quality
        near=0.5,
        far=6.0,
        aspect_scales=aspect_scales,
    )
    print(f"  Dataset: {len(dataset)} views, image size {H}×{W}")

    # Quick stats
    all_max = max(d['image'].max().item() for d in dataset)
    all_mean = np.mean([d['image'].mean().item() for d in dataset])
    print(f"  MIP intensity: mean={all_mean:.4f}, global_max={all_max:.4f}")

    # ------------------------------------------------------------------
    #  6. Train splatting renderer against MIP ground truth
    # ------------------------------------------------------------------
    print("\nStarting splatting training...")
    trainer = SplattingTrainer(
        means=means,
        log_scales=log_scales,
        quaternions=quaternions,
        log_amplitudes=log_amplitudes,
        lr=1e-3,
        lambda_ssim=0.0,
        pixels_per_step=8192,
        aspect_scales=aspect_scales,
    )

    save_template = os.path.join(
        os.path.dirname(__file__), "..",
        "checkpoints", "splat_step{step}.pt"
    )

    history = trainer.train(
        camera=camera,
        dataset=dataset,
        n_steps=10000,
        log_every=100,
        save_path=save_template,
        save_every=2000,
    )

    # ------------------------------------------------------------------
    #  7. Final validation: render from a few viewpoints
    # ------------------------------------------------------------------
    print("\nFinal validation renders...")
    gaussians = trainer._build_gaussians_corrected()
    renderer = GaussianSplattingRenderer(use_alpha_compositing=True)

    for view_idx in [0, len(dataset)//4, len(dataset)//2]:
        view = dataset[view_idx]
        result = renderer.render(gaussians, camera, view['R'], view['T'])
        gt = view['image']
        pred = result['image'].squeeze(-1)  # (H, W)
        l1_err = F.l1_loss(pred, gt).item()
        print(f"  View {view_idx} (el={view['elevation']:.0f}°, "
              f"az={view['azimuth']:.0f}°): "
              f"L1={l1_err:.4f}, visible={result['n_visible']}")

    print("\n✓ Splatting training pipeline complete!")

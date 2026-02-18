"""
Gaussian Splatting Rendering Module
====================================
Implements the 2D splatting pipeline for rendering 3D Gaussians,
plus multi-view MIP generation for training supervision.
"""

from .rendering import (
    Camera,
    GaussianParameters,
    GaussianSplattingRenderer,
    SplattingTrainer,
    transform_to_camera,
    compute_projection_jacobian,
    project_to_2d,
    evaluate_gaussian_2d,
    evaluate_gaussian_2d_batched,
    splat_gaussians,
    splat_gaussians_alpha,
    splatting_loss,
    ssim_loss,
    gaussians_from_model,
    orbit_camera_pose,
    compute_aspect_scales,
    apply_aspect_correction,
    load_volume,
    render_mip_from_camera,
    generate_camera_poses,
    generate_mip_dataset,
)

__all__ = [
    "Camera",
    "GaussianParameters",
    "GaussianSplattingRenderer",
    "SplattingTrainer",
    "transform_to_camera",
    "compute_projection_jacobian",
    "project_to_2d",
    "evaluate_gaussian_2d",
    "evaluate_gaussian_2d_batched",
    "splat_gaussians",
    "splat_gaussians_alpha",
    "splatting_loss",
    "ssim_loss",
    "gaussians_from_model",
    "orbit_camera_pose",
    "compute_aspect_scales",
    "apply_aspect_correction",
    "load_volume",
    "render_mip_from_camera",
    "generate_camera_poses",
    "generate_mip_dataset",
]

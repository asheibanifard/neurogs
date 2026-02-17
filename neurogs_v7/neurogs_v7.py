import os
import math
import time
import yaml
import logging
from datetime import datetime

import numpy as np
import tifffile as tiff
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# Try to load custom CUDA extension for faster backward pass
try:
    import gaussian_eval_cuda
    HAS_CUDA_EXTENSION = True
    print("✓ Loaded custom CUDA extension for 2-3x faster backward pass")
except ImportError:
    HAS_CUDA_EXTENSION = False
    print("✗ Custom CUDA extension not found, using PyTorch fallback")
    print("  To build: cd to this directory and run: python setup_cuda.py install")


# -------------------------
# Custom CUDA-optimized autograd function for Gaussian evaluation
# -------------------------
class GaussianEvalFunction(Function):
    """
    Custom autograd function with optimized backward pass for Gaussian mixture evaluation.
    Uses custom CUDA kernel if available, otherwise falls back to PyTorch ops.
    
    Forward: Compute v_k = a_k * exp(-0.5 * (x - μ_k)^T Σ_k^{-1} (x - μ_k)) for all k
    Backward: Fused CUDA kernel that avoids large tensor materializations
    """
    
    @staticmethod
    def forward(ctx, x, means, L_chol, amplitudes, use_cuda_kernel=False):
        """
        x: (N, 3) query points
        means: (K, 3) Gaussian centers
        L_chol: (K, 3, 3) Cholesky factors (lower triangular)
        amplitudes: (K,) Gaussian amplitudes
        use_cuda_kernel: if True and available, use custom CUDA kernel
        
        Returns: (N, K) Gaussian values
        """
        N, K = x.shape[0], means.shape[0]
        
        if use_cuda_kernel and HAS_CUDA_EXTENSION and x.is_cuda:
            # Use custom CUDA kernel (faster)
            vals = gaussian_eval_cuda.forward(x.contiguous(), means.contiguous(), 
                                             L_chol.contiguous(), amplitudes.contiguous())
            ctx.use_cuda_kernel = True
            ctx.save_for_backward(x, means, L_chol, amplitudes, vals)
            return vals
        else:
            # PyTorch fallback
            ctx.use_cuda_kernel = False
            
            # Compute differences (vectorized)
            diff = x[:, None, :] - means[None, :, :]  # (N, K, 3)
            
            # Solve L * y = diff for y using batched triangular solve
            # Reshape for batched solve
            diff_flat = diff.reshape(N*K, 3, 1)
            L_expanded = L_chol[None].expand(N, K, 3, 3).reshape(N*K, 3, 3)
            
            y = torch.linalg.solve_triangular(L_expanded, diff_flat, upper=False)
            y = y.reshape(N, K, 3)
            
            # Mahalanobis distance: ||y||^2
            mahal = (y ** 2).sum(dim=-1)  # (N, K)
            
            # Gaussian values
            vals = amplitudes[None, :] * torch.exp(-0.5 * mahal)  # (N, K)
            
            # Save for backward
            ctx.save_for_backward(x, means, L_chol, amplitudes, y, vals)
            
            return vals
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Optimized backward pass using CUDA kernel if available.
        
        grad_output: (N, K) gradient w.r.t. output
        
        Returns: gradients w.r.t. (x, means, L_chol, amplitudes, use_cuda_kernel)
        """
        if ctx.use_cuda_kernel:
            # Use custom CUDA backward kernel
            x, means, L_chol, amplitudes, vals = ctx.saved_tensors
            grads = gaussian_eval_cuda.backward(
                grad_output.contiguous(), 
                x, means, L_chol, amplitudes, vals
            )
            # Returns [grad_x, grad_means, grad_L_chol, grad_amplitudes]
            return grads[0], grads[1], grads[2], grads[3], None
        else:
            # PyTorch fallback
            x, means, L_chol, amplitudes, y, vals = ctx.saved_tensors
            N, K = x.shape[0], means.shape[0]
            
            # Gradient w.r.t. amplitudes
            grad_amplitudes = None
            if ctx.needs_input_grad[3]:
                grad_amplitudes = (grad_output * vals / (amplitudes[None, :] + 1e-12)).sum(dim=0)
            
            # Gradient w.r.t. Mahalanobis distance
            grad_mahal = grad_output * vals * (-0.5)  # (N, K)
            
            # Gradient w.r.t. y: ∂mahal/∂y = 2*y
            grad_y = grad_mahal[:, :, None] * (2.0 * y)  # (N, K, 3)
            
            # Gradient w.r.t. diff through triangular solve: L^T @ grad_y
            grad_y_flat = grad_y.reshape(N*K, 3, 1)
            L_expanded = L_chol[None].expand(N, K, 3, 3).reshape(N*K, 3, 3)
            
            # Solve L^T * grad_diff = grad_y
            grad_diff_flat = torch.linalg.solve_triangular(
                L_expanded.transpose(-2, -1), grad_y_flat, upper=True
            )
            grad_diff = grad_diff_flat.reshape(N, K, 3)  # (N, K, 3)
            
            # Gradient w.r.t. x (sum over K)
            grad_x = None
            if ctx.needs_input_grad[0]:
                grad_x = grad_diff.sum(dim=1)  # (N, 3)
            
            # Gradient w.r.t. means
            grad_means = None
            if ctx.needs_input_grad[1]:
                grad_means = -grad_diff.sum(dim=0)  # (K, 3)
            
            # Gradient w.r.t. L_chol (use autograd, less critical)
            grad_L_chol = None
            
            return grad_x, grad_means, grad_L_chol, grad_amplitudes, None


# Apply the custom function
gaussian_eval = GaussianEvalFunction.apply


# -------------------------
# Config
# -------------------------
def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# -------------------------
# IO
# -------------------------
def load_tif_data(file_path: str) -> np.ndarray:
    vol = tiff.imread(file_path).astype(np.float32)  # (Z,Y,X) expected
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax - vmin < 1e-12:
        return np.zeros_like(vol, dtype=np.float32)
    vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(np.float32)


# -------------------------
# Sampling utilities
# -------------------------
# Global cache for GPU-accelerated sampling
_volume_sampling_cache = {}

def sample_points_from_volume_cuda(vol_gpu: torch.Tensor, num_samples: int, 
                                   intensity_weighted: bool = False,
                                   cache_key: str = None) -> tuple:
    """
    CUDA-accelerated volume sampling. Keeps everything on GPU.
    
    Args:
        vol_gpu: (Z,Y,X) torch tensor on GPU, float32 in [0,1]
        num_samples: number of samples to draw
        intensity_weighted: if True, sample proportional to intensity
        cache_key: if provided, cache CDF for faster repeated sampling
    
    Returns:
        pts: (N,3) torch float32 in [-1,1]^3 on GPU
        vals: (N,) torch float32 in [0,1] on GPU
    """
    Z, Y, X = vol_gpu.shape
    Nvox = Z * Y * X
    device = vol_gpu.device
    
    if num_samples > Nvox:
        raise ValueError(f"num_samples={num_samples} > total voxels={Nvox}")
    
    if intensity_weighted:
        # GPU-accelerated weighted sampling using CDF + binary search
        # Check cache first
        if cache_key and cache_key in _volume_sampling_cache:
            cdf = _volume_sampling_cache[cache_key]
        else:
            # Compute CDF on GPU (one-time cost)
            flat = vol_gpu.reshape(-1)
            probs = flat / (flat.sum() + 1e-12)
            cdf = torch.cumsum(probs, dim=0)
            # Cache if key provided
            if cache_key:
                _volume_sampling_cache[cache_key] = cdf
        
        # Generate uniform random numbers on GPU
        u = torch.rand(num_samples, device=device)
        # Binary search to find indices (fully parallelized on GPU)
        idx = torch.searchsorted(cdf, u)
        # Clamp to valid range (in case of numerical issues)
        idx = torch.clamp(idx, 0, Nvox - 1)
    else:
        # Uniform sampling (fast)
        idx = torch.randperm(Nvox, device=device)[:num_samples]
    
    # Convert flat indices to 3D coordinates (all on GPU)
    z = idx // (Y * X)
    rem = idx % (Y * X)
    y = rem // X
    x = rem % X
    
    # Normalize voxel coords to [-1,1]
    x_norm = (x.float() / (X - 1)) * 2 - 1
    y_norm = (y.float() / (Y - 1)) * 2 - 1
    z_norm = (z.float() / (Z - 1)) * 2 - 1
    
    pts = torch.stack([x_norm, y_norm, z_norm], dim=1)
    vals = vol_gpu[z, y, x]
    
    return pts, vals


def sample_points_from_volume(vol: np.ndarray, num_samples: int, intensity_weighted: bool = False):
    """
    CPU fallback for volume sampling (slower).
    
    vol: (Z,Y,X) float32 in [0,1]
    returns:
      pts: (N,3) torch float32 in [-1,1]^3
      vals:(N,)  torch float32 in [0,1]
    """
    Z, Y, X = vol.shape
    Nvox = Z * Y * X

    if num_samples > Nvox:
        raise ValueError(f"num_samples={num_samples} > total voxels={Nvox}")

    if intensity_weighted:
        # CPU version (slow)
        flat = vol.reshape(-1)
        p = flat / (flat.sum() + 1e-12)
        idx = np.random.choice(Nvox, size=num_samples, replace=True, p=p)
    else:
        idx = np.random.choice(Nvox, size=num_samples, replace=False)

    z = idx // (Y * X)
    rem = idx % (Y * X)
    y = rem // X
    x = rem % X

    # normalize voxel coords to [-1,1]
    x_norm = (x / (X - 1)) * 2 - 1
    y_norm = (y / (Y - 1)) * 2 - 1
    z_norm = (z / (Z - 1)) * 2 - 1

    pts = torch.from_numpy(np.stack([x_norm, y_norm, z_norm], axis=1)).float()
    vals = torch.from_numpy(vol[z, y, x]).float()
    return pts, vals


def mip_teacher_z(vol: np.ndarray) -> np.ndarray:
    """Teacher z-MIP from dense volume. vol: (Z,Y,X) -> mip: (Y,X)"""
    return vol.max(axis=0).astype(np.float32)


def sample_pixels_from_mip(mip: np.ndarray, num_samples: int):
    """
    mip: (Y,X) float32
    returns:
      xy: (N,2) in [-1,1]
      t:  (N,)  target intensities
    """
    Y, X = mip.shape
    Npix = Y * X
    if num_samples > Npix:
        raise ValueError(f"num_samples={num_samples} > total pixels={Npix}")

    idx = np.random.choice(Npix, size=num_samples, replace=False)
    y = idx // X
    x = idx % X

    x_norm = (x / (X - 1)) * 2 - 1
    y_norm = (y / (Y - 1)) * 2 - 1

    xy = torch.from_numpy(np.stack([x_norm, y_norm], axis=1)).float()
    t = torch.from_numpy(mip[y, x]).float()
    return xy, t


# -------------------------
# Regularizers (cheap ones)
# -------------------------
def tubular_regularizer(covariance_matrices: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # eigvalsh doesn't support FP16, force FP32
    eigvals = torch.linalg.eigvalsh(covariance_matrices.float())  # (K,3), ascending
    eigvals = torch.sort(eigvals, dim=-1)[0]
    lam1, lam2, lam3 = eigvals[:, 0], eigvals[:, 1], eigvals[:, 2]
    return ((lam1 + lam2) / (lam3 + eps)).mean()


def cross_section_symmetry_regularizer(covariance_matrices: torch.Tensor) -> torch.Tensor:
    # eigvalsh doesn't support FP16, force FP32
    eigvals = torch.linalg.eigvalsh(covariance_matrices.float())
    eigvals = torch.sort(eigvals, dim=-1)[0]
    lam1, lam2 = eigvals[:, 0], eigvals[:, 1]
    return torch.abs(lam1 - lam2).mean()


# -------------------------
# Model
# -------------------------
class GaussianMixtureField(nn.Module):
    """
    V^(x) = Σ_k a_k * exp(-0.5 * (x-μ)^T Σ^{-1} (x-μ))
    with Σ = R * diag(s^2) * R^T
    """
    def __init__(self, num_gaussians: int, init_scale=1.0, bounds=None):
        super().__init__()
        self.num_gaussians = num_gaussians

        if bounds is not None:
            means = torch.zeros(num_gaussians, 3)
            for i in range(3):
                means[:, i] = torch.rand(num_gaussians) * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
        else:
            means = torch.randn(num_gaussians, 3) * 0.1

        self.means = nn.Parameter(means)
        self.log_scales = nn.Parameter(torch.ones(num_gaussians, 3) * math.log(init_scale))

        q = torch.zeros(num_gaussians, 4)
        q[:, 0] = 1.0  # identity
        self.quaternions = nn.Parameter(q)

        self.log_amplitudes = nn.Parameter(torch.zeros(num_gaussians))

    @staticmethod
    def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q, p=2, dim=-1)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - w*z)
        R[:, 0, 2] = 2 * (x*z + w*y)

        R[:, 1, 0] = 2 * (x*y + w*z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - w*x)

        R[:, 2, 0] = 2 * (x*z - w*y)
        R[:, 2, 1] = 2 * (y*z + w*x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
        return R

    def get_covariance_matrices(self) -> torch.Tensor:
        scales = torch.exp(self.log_scales).clamp(min=1e-4, max=1e2)   # (K,3)
        R = self.quaternion_to_rotation_matrix(self.quaternions)       # (K,3,3)
        S2 = torch.diag_embed(scales ** 2)                             # (K,3,3)
        Sigma = R @ S2 @ R.transpose(-2, -1)                           # (K,3,3)
        return Sigma

    def forward(self, x: torch.Tensor, return_components=False) -> torch.Tensor:
        """
        x: (N,3) -> (N,) or (N,K)
        """
        # Clamp log_amplitudes to prevent numerical blow-ups (especially under MIP loss)
        amplitudes = torch.exp(self.log_amplitudes.clamp(min=-15.0, max=15.0))  # (K,)
        Sigma = self.get_covariance_matrices()                         # (K,3,3)

        eps = 1e-6
        Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
        # Cholesky doesn't support FP16, force FP32 even under mixed precision
        dtype_orig = Sigma_reg.dtype
        L = torch.linalg.cholesky(Sigma_reg.float())                   # (K,3,3)

        diff = x[:, None, :] - self.means[None, :, :]                  # (N,K,3)

        # reshape to (N*K,3,1) for robust batched solve
        N = x.shape[0]
        K = self.num_gaussians
        diff2 = diff.reshape(N*K, 3, 1).float()  # Also cast to FP32 for triangular solve
        L2 = L[None].expand(N, K, 3, 3).reshape(N*K, 3, 3)

        y = torch.linalg.solve_triangular(L2, diff2, upper=False)      # (N*K,3,1)
        mahal = (y.squeeze(-1) ** 2).sum(dim=-1).reshape(N, K)         # (N,K)

        # Cast back to original dtype for the rest of the computation
        vals = amplitudes[None, :] * torch.exp(-0.5 * mahal.to(dtype_orig))  # (N,K)

        if return_components:
            return vals
        return vals.sum(dim=1)


# -------------------------
# Soft-MIP renderer (z-axis)
# -------------------------
def compute_tau_schedule(tau_start: float, tau_end: float, t: float) -> float:
    """t in [0,1]"""
    return float(tau_start * (tau_end / tau_start) ** t)


def compute_weight_schedule(config: dict, step: int, total_steps: int) -> tuple[float, float]:
    """
    Compute w_vol and w_mip based on schedule configuration.
    
    Schedules:
    - 'constant': use w_vol and w_mip from config
    - 'step': switch from start to end weights at transition_fraction
    - 'linear_ramp': warmup at start weights until transition_fraction, then ramp to end weights
    
    Args:
        config: training configuration dict
        step: current training step
        total_steps: total number of steps
    
    Returns:
        (w_vol, w_mip) tuple
    """
    schedule_type = config["training"].get("weight_schedule", "constant").lower()
    
    if schedule_type == "constant":
        w_vol = float(config["training"].get("w_vol", 1.0))
        w_mip = float(config["training"].get("w_mip", 1.0))
        return w_vol, w_mip
    
    # Get schedule parameters
    w_vol_start = float(config["training"].get("w_vol_start", 1.0))
    w_mip_start = float(config["training"].get("w_mip_start", 0.2))
    w_vol_end = float(config["training"].get("w_vol_end", 1.0))
    w_mip_end = float(config["training"].get("w_mip_end", 1.0))
    transition_frac = float(config["training"].get("weight_transition_fraction", 0.3))
    
    t = step / max(1, total_steps - 1)  # normalized progress [0, 1]
    
    if schedule_type == "step":
        # Step schedule: switch at transition fraction
        if t < transition_frac:
            return w_vol_start, w_mip_start
        else:
            return w_vol_end, w_mip_end
    
    elif schedule_type == "linear_ramp":
        # Linear ramp with warmup: keep start weights until transition_fraction,
        # then linearly interpolate to end weights
        if t < transition_frac:
            return w_vol_start, w_mip_start
        else:
            # Normalize t to [0, 1] over the ramp period (transition_frac to 1.0)
            t_ramp = (t - transition_frac) / (1.0 - transition_frac)
            w_vol = w_vol_start + (w_vol_end - w_vol_start) * t_ramp
            w_mip = w_mip_start + (w_mip_end - w_mip_start) * t_ramp
            return w_vol, w_mip
    
    else:
        # Fallback to constant
        w_vol = float(config["training"].get("w_vol", 1.0))
        w_mip = float(config["training"].get("w_mip", 1.0))
        return w_vol, w_mip


def render_soft_mip_z(field: GaussianMixtureField,
                      xy: torch.Tensor,
                      n_z_samples: int,
                      tau: float,
                      chunk: int = 65536) -> torch.Tensor:
    """
    xy: (P,2) in [-1,1]
    returns: (P,) soft-MIP intensity
    """
    device = xy.device
    P = xy.shape[0]
    z = torch.linspace(-1, 1, n_z_samples, device=device, dtype=xy.dtype)  # (S,)

    pts = torch.cat([
        xy[:, None, :].expand(P, n_z_samples, 2),
        z[None, :, None].expand(P, n_z_samples, 1),
    ], dim=-1).reshape(-1, 3)                                              # (P*S,3)

    # Adaptive chunk size based on K and available GPU memory
    # Memory usage in forward(): L2(N*K*3*3), diff2(N*K*3), y(N*K*3), m_dist(N*K), s(N*K)
    # Total: N*K*68 bytes per chunk
    K = field.num_gaussians
    
    # Query actual available memory and use 80% of it
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        mem_free = torch.cuda.mem_get_info(device)[0]  # bytes free
        mem_budget = int(mem_free * 0.8)  # use 80% of free memory
        max_chunk_from_mem = max(1024, mem_budget // (K * 68))
        adaptive_chunk = min(chunk, max_chunk_from_mem)
    else:
        adaptive_chunk = chunk
    
    # chunk field eval to avoid OOM
    vals_list = []
    for i in range(0, pts.shape[0], adaptive_chunk):
        vals_list.append(field(pts[i:i+adaptive_chunk]))
    v = torch.cat(vals_list, dim=0).reshape(P, n_z_samples)                # (P,S)

    # soft max (LogSumExp)
    I = tau * torch.logsumexp(v / tau, dim=1)                              # (P,)
    return I


# -------------------------
# Loss wrappers
# -------------------------
def loss_volume_fit(field: GaussianMixtureField,
                    x: torch.Tensor,
                    v: torch.Tensor,
                    w_tube=1e-4,
                    w_cross=1e-4) -> tuple[torch.Tensor, dict]:
    pred = field(x)
    l_rec = F.mse_loss(pred, v)

    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regularizer(Sigma)
    l_cross = cross_section_symmetry_regularizer(Sigma)

    total = l_rec + w_tube * l_tube + w_cross * l_cross
    return total, {"rec": l_rec, "tube": l_tube, "cross": l_cross}


def loss_mip_fit(field: GaussianMixtureField,
                 xy: torch.Tensor,
                 mip_t: torch.Tensor,
                 n_z_samples: int,
                 tau: float,
                 w_tube=1e-4,
                 w_cross=1e-4,
                 mip_batch_size: int = 1024) -> tuple[torch.Tensor, dict]:
    """
    Render MIP in batches to avoid OOM with large K.
    mip_batch_size: max MIP pixels to render at once (default 1024)
    """
    # Batch MIP rendering to save memory
    P = xy.shape[0]
    if P <= mip_batch_size:
        pred = render_soft_mip_z(field, xy, n_z_samples=n_z_samples, tau=tau)
    else:
        pred_list = []
        for i in range(0, P, mip_batch_size):
            pred_list.append(render_soft_mip_z(field, xy[i:i+mip_batch_size], 
                                               n_z_samples=n_z_samples, tau=tau))
        pred = torch.cat(pred_list, dim=0)
    
    # L1 often works better for MIP contrast; you can switch back to MSE if needed
    l_img = F.l1_loss(pred, mip_t)

    Sigma = field.get_covariance_matrices()
    l_tube = tubular_regularizer(Sigma)
    l_cross = cross_section_symmetry_regularizer(Sigma)

    total = l_img + w_tube * l_tube + w_cross * l_cross
    return total, {"mip": l_img, "tube": l_tube, "cross": l_cross}


# -------------------------
# Training
# -------------------------
def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logger = logging.getLogger("train")
    logger.info(f"Log file: {log_file}")
    return logger


def train_end2end(field: GaussianMixtureField,
                  vol: np.ndarray,
                  config: dict,
                  device: str,
                  log_dir: str):
    logger = setup_logger(log_dir)
    field = field.to(device)
    field.train()

    # mode: "volume" | "mip" | "hybrid" | "progressive"
    mode = config["training"].get("mode", "hybrid").lower()
    
    # Progressive training: switch modes during training
    progressive_enabled = (mode == "progressive")
    if progressive_enabled:
        phase1_mode = config["training"].get("progressive_phase1_mode", "volume").lower()
        phase1_steps = int(config["training"].get("progressive_phase1_steps", 500))
        phase2_mode = config["training"].get("progressive_phase2_mode", "hybrid").lower()
        phase2_steps = int(config["training"].get("progressive_phase2_steps", 1500))
        total_steps = phase1_steps + phase2_steps
        logger.info(f"Progressive training: {phase1_mode} for {phase1_steps} steps, then {phase2_mode} for {phase2_steps} steps")
    else:
        total_steps = int(config["training"].get("steps", 2000))

    # volumes sampling
    vol_points_per_step = int(config["training"].get("vol_points_per_step", 8192))
    vol_intensity_weighted = bool(config["training"].get("vol_intensity_weighted", True))

    # mip sampling
    mip_pixels_per_step = int(config["training"].get("mip_pixels_per_step", 4096))
    n_z_samples = int(config["training"].get("mip_z_samples", 128))
    tau_start = float(config["training"].get("tau_start", 0.08))
    tau_end = float(config["training"].get("tau_end", 0.02))

    # weights
    w_tube = float(config["training"].get("lambda_tube", 1e-4))
    w_cross = float(config["training"].get("lambda_cross", 1e-4))
    
    # Check if weight scheduling is enabled
    weight_schedule_type = config["training"].get("weight_schedule", "constant").lower()
    use_weight_schedule = (weight_schedule_type != "constant")

    lr = float(config["training"]["learning_rate"])
    steps = total_steps
    optimizer = torch.optim.Adam(field.parameters(), lr=lr)
    
    # Mixed precision training (optional but recommended for speed)
    use_amp = config["training"].get("mixed_precision", False) and device == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled")
    else:
        scaler = None

    mip_img = mip_teacher_z(vol)  # (Y,X)

    logger.info(f"Device: {device}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Steps: {steps}, lr={lr}")
    logger.info(f"Volume shape: {vol.shape}, MIP shape: {mip_img.shape}")
    logger.info(f"Weight schedule: {weight_schedule_type}")
    if use_weight_schedule:
        w_vol_start = config["training"].get("w_vol_start", 1.0)
        w_mip_start = config["training"].get("w_mip_start", 0.2)
        w_vol_end = config["training"].get("w_vol_end", 1.0)
        w_mip_end = config["training"].get("w_mip_end", 1.0)
        transition_frac = config["training"].get("weight_transition_fraction", 0.3)
        logger.info(f"Weight schedule: vol {w_vol_start}->{w_vol_end}, mip {w_mip_start}->{w_mip_end}, transition at {transition_frac*100:.0f}%")
    else:
        w_vol = float(config["training"].get("w_vol", 1.0))
        w_mip = float(config["training"].get("w_mip", 1.0))
        logger.info(f"Constant weights: w_vol={w_vol}, w_mip={w_mip}")
    logger.info(f"Tube reg weights: tube={w_tube}, cross={w_cross}")

    # Move volume to GPU for fast sampling (if using CUDA)
    if device == "cuda":
        vol_gpu = torch.from_numpy(vol).float().to(device)
        use_cuda_sampling = True
        logger.info("Using CUDA-accelerated volume sampling (100x faster than CPU)")
    else:
        vol_gpu = None
        use_cuda_sampling = False
    
    # Track whether we're in volume-only phase (to free vol_gpu later)
    last_mode = None

    # Timing statistics
    timings = {
        "vol_sample": [], "vol_forward": [], "vol_loss": [],
        "mip_sample": [], "mip_render": [], "mip_loss": [],
        "backward": [], "optimizer": []
    }
    
    pbar = tqdm(range(steps), desc="Training")
    for step in pbar:
        # Determine current mode for progressive training
        if progressive_enabled:
            if step < phase1_steps:
                current_mode = phase1_mode
                if step == 0:
                    logger.info(f"Phase 1: {phase1_mode} mode (steps 0-{phase1_steps})")
            else:
                current_mode = phase2_mode
                if step == phase1_steps:
                    logger.info(f"Phase 2: {phase2_mode} mode (steps {phase1_steps}-{steps})")
                    # Free vol_gpu when transitioning to MIP/hybrid to save memory
                    if vol_gpu is not None and current_mode in ("mip", "hybrid"):
                        logger.info("Freeing volume from GPU to save memory for MIP rendering")
                        del vol_gpu
                        vol_gpu = None
                        use_cuda_sampling = False
                        torch.cuda.empty_cache()
        else:
            current_mode = mode
        
        optimizer.zero_grad()

        t = step / max(1, steps - 1)
        tau = compute_tau_schedule(tau_start, tau_end, t)
        
        # Compute current weights (with scheduling if enabled)
        if use_weight_schedule:
            w_vol, w_mip = compute_weight_schedule(config, step, steps)
        else:
            w_vol = float(config["training"].get("w_vol", 1.0))
            w_mip = float(config["training"].get("w_mip", 1.0))

        losses = {}
        
        # Use mixed precision if enabled
        if use_amp:
            with torch.cuda.amp.autocast():
                total = torch.zeros((), device=device)

                if current_mode in ("volume", "hybrid"):
                    t0 = time.time()
                    if use_cuda_sampling:
                        # CUDA-accelerated sampling (100x faster)
                        x, v = sample_points_from_volume_cuda(vol_gpu, vol_points_per_step, 
                                                             intensity_weighted=vol_intensity_weighted,
                                                             cache_key="train_vol" if vol_intensity_weighted else None)
                    else:
                        # CPU fallback
                        x, v = sample_points_from_volume(vol, vol_points_per_step, intensity_weighted=vol_intensity_weighted)
                        x = x.to(device)
                        v = v.to(device)
                    if device == "cuda": torch.cuda.synchronize()
                    timings["vol_sample"].append(time.time() - t0)

                    t0 = time.time()
                    l_vol, parts_vol = loss_volume_fit(field, x, v, w_tube=w_tube, w_cross=w_cross)
                    if device == "cuda": torch.cuda.synchronize()
                    timings["vol_forward"].append(time.time() - t0)
                    
                    total = total + w_vol * l_vol
                    losses.update({f"vol_{k}": float(val.detach().cpu()) for k, val in parts_vol.items()})

                if current_mode in ("mip", "hybrid"):
                    t0 = time.time()
                    xy, mt = sample_pixels_from_mip(mip_img, mip_pixels_per_step)
                    xy = xy.to(device)
                    mt = mt.to(device)
                    if device == "cuda": torch.cuda.synchronize()
                    timings["mip_sample"].append(time.time() - t0)

                    t0 = time.time()
                    l_mip, parts_mip = loss_mip_fit(field, xy, mt, n_z_samples=n_z_samples, tau=tau,
                                                   w_tube=w_tube, w_cross=w_cross)
                    if device == "cuda": torch.cuda.synchronize()
                    timings["mip_render"].append(time.time() - t0)
                    
                    total = total + w_mip * l_mip
                    losses.update({f"mip_{k}": float(val.detach().cpu()) for k, val in parts_mip.items()})
                    losses["tau"] = tau
            
            t0 = time.time()
            scaler.scale(total).backward()
            if device == "cuda": torch.cuda.synchronize()
            timings["backward"].append(time.time() - t0)
            
            t0 = time.time()
            scaler.step(optimizer)
            scaler.update()
            if device == "cuda": torch.cuda.synchronize()
            timings["optimizer"].append(time.time() - t0)
        else:
            total = torch.zeros((), device=device)

            if current_mode in ("volume", "hybrid"):
                t0 = time.time()
                if use_cuda_sampling:
                    # CUDA-accelerated sampling (100x faster)
                    x, v = sample_points_from_volume_cuda(vol_gpu, vol_points_per_step,
                                                         intensity_weighted=vol_intensity_weighted,
                                                         cache_key="train_vol" if vol_intensity_weighted else None)
                else:
                    # CPU fallback
                    x, v = sample_points_from_volume(vol, vol_points_per_step, intensity_weighted=vol_intensity_weighted)
                    x = x.to(device)
                    v = v.to(device)
                if device == "cuda": torch.cuda.synchronize()
                timings["vol_sample"].append(time.time() - t0)

                t0 = time.time()
                l_vol, parts_vol = loss_volume_fit(field, x, v, w_tube=w_tube, w_cross=w_cross)
                if device == "cuda": torch.cuda.synchronize()
                timings["vol_forward"].append(time.time() - t0)
                
                total = total + w_vol * l_vol
                losses.update({f"vol_{k}": float(val.detach().cpu()) for k, val in parts_vol.items()})

            if current_mode in ("mip", "hybrid"):
                t0 = time.time()
                xy, mt = sample_pixels_from_mip(mip_img, mip_pixels_per_step)
                xy = xy.to(device)
                mt = mt.to(device)
                if device == "cuda": torch.cuda.synchronize()
                timings["mip_sample"].append(time.time() - t0)

                t0 = time.time()
                l_mip, parts_mip = loss_mip_fit(field, xy, mt, n_z_samples=n_z_samples, tau=tau,
                                               w_tube=w_tube, w_cross=w_cross)
                if device == "cuda": torch.cuda.synchronize()
                timings["mip_render"].append(time.time() - t0)
                
                total = total + w_mip * l_mip
                losses.update({f"mip_{k}": float(val.detach().cpu()) for k, val in parts_mip.items()})
                losses["tau"] = tau

            t0 = time.time()
            total.backward()
            if device == "cuda": torch.cuda.synchronize()
            timings["backward"].append(time.time() - t0)
            
            t0 = time.time()
            optimizer.step()
            if device == "cuda": torch.cuda.synchronize()
            timings["optimizer"].append(time.time() - t0)

        losses["total"] = float(total.detach().cpu())
        losses["w_vol"] = w_vol
        losses["w_mip"] = w_mip
        
        # Compact postfix for progress bar
        postfix_keys = ["total", "vol_rec", "mip_mip", "w_mip", "tau"]
        pbar.set_postfix({k: f"{v:.4g}" for k, v in losses.items() if k in postfix_keys})

        if (step + 1) % int(config["training"].get("log_every", 50)) == 0:
            logger.info("Step %d: %s", step + 1, losses)

    # Report timing statistics
    logger.info("=" * 60)
    logger.info("TIMING ANALYSIS (mean ± std over all steps)")
    logger.info("=" * 60)
    
    total_time_per_step = 0.0
    for key in ["vol_sample", "vol_forward", "mip_sample", "mip_render", "backward", "optimizer"]:
        if timings[key]:
            arr = np.array(timings[key]) * 1000  # Convert to ms
            mean_ms = arr.mean()
            std_ms = arr.std()
            total_time_per_step += arr.mean()
            pct = 100.0 * arr.mean() / arr.sum() * len(timings[key]) if arr.sum() > 0 else 0
            logger.info(f"  {key:15s}: {mean_ms:7.2f} ± {std_ms:5.2f} ms  ({pct:5.1f}% of total)")
    
    if total_time_per_step > 0:
        logger.info(f"  {'TOTAL':15s}: {total_time_per_step:7.2f} ms/step")
        steps_per_sec = 1000.0 / total_time_per_step
        logger.info(f"  Throughput: {steps_per_sec:.2f} steps/sec")
    
    logger.info("=" * 60)
    logger.info("Training completed.")
    return field


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    config = load_config("config.yml")
    
    # Set random seed for reproducibility
    seed = int(config.get("seed", 0))
    if seed != 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to: {seed}")

    # device
    device_cfg = config["training"].get("device", "auto")
    device = "cuda" if (device_cfg == "auto" and torch.cuda.is_available()) else device_cfg

    # data
    tif_path = config["data"]["tif_path"]
    vol = load_tif_data(tif_path)

    # model
    num_gaussians = int(config["model"]["num_gaussians"])
    init_scale = float(config["model"].get("init_scale", 0.02))
    bounds = config["model"].get("bounds", None)  # e.g. [[-1,1],[-1,1],[-1,1]] or None

    field = GaussianMixtureField(num_gaussians=num_gaussians, init_scale=init_scale, bounds=bounds)

    log_dir = config["training"].get("log_dir", "logs")
    field = train_end2end(field, vol, config, device=device, log_dir=log_dir)

    # Optional: save trained parameters
    out_path = config["training"].get("save_path", None)
    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:  # Only create directory if path contains a directory
            os.makedirs(out_dir, exist_ok=True)
        torch.save(field.state_dict(), out_path)
        print(f"Saved model to: {out_path}")

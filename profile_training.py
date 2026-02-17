#!/usr/bin/env python3
"""
Training Profiler - Identify bottlenecks in NeuroGS training
=============================================================
"""

import os
import time
import torch
import numpy as np
import tifffile as tiff
from collections import defaultdict
from contextlib import contextmanager

# Import from training script
from train_standalone import (
    DEVICE, TRAINING_CONFIG, TIF_PATH, VOXEL_SPACING,
    make_coord_grid_zyx, make_neurite_map,
    init_gaussians_from_neurite_map, sample_points,
    precompute_sampling_cdf, mse_loss, unweighted_mse,
    charbonnier, LaplaceEntropyModel, QuantSteps, ste_round,
    patch_topology_loss, reconstruction_tv_loss,
    patch_ssim_loss, edge_aware_loss,
    field_grad_smoothness, smoothness_loss, sparsity_loss,
    overlap_loss_mahalanobis
)

try:
    from cuda_ops import CUDAGaussianMixtureVolume, CUDA_AVAILABLE
    USE_CUDA_OPS = CUDA_AVAILABLE
except ImportError:
    USE_CUDA_OPS = False
    from train_standalone import GaussianMixtureVolume


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        torch.cuda.synchronize()
        elapsed = time.time() - self.start
        self.stats[self.name].append(elapsed * 1000)  # ms


def profile_training_iteration(model, V, coords_grid, M, cfg, H_mu, H_logs, H_q, H_a, H_b, Q, 
                                opt, scaler, use_amp=True, it=0):
    """Profile a single training iteration"""
    
    stats = defaultdict(list)
    
    # Phase logic
    phase1_end = int(cfg["steps"] * cfg["phase1_end_frac"])
    phase2_end = int(cfg["steps"] * cfg["phase2_end_frac"])
    if it < phase1_end:
        phase_num = 1
        reg_scale = 1.0
    elif it < phase2_end:
        phase_num = 2
        reg_scale = 1.0
    else:
        phase_num = 3
        reg_scale = cfg["phase3_reg_scale"]
    
    # Precompute sampling
    sampling_cdf, sampling_cand = precompute_sampling_cdf(M)
    
    with Timer("1_sampling", stats):
        n_u = int(cfg["batch"] * 0.6)
        n_b = cfg["batch"] - n_u
        pts, tgt, mval = sample_points(coords_grid, V, M, n_u, n_b,
                                       cdf=sampling_cdf, cdf_cand=sampling_cand)
    
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        with Timer("2_forward", stats):
            pred = model(pts)
        
        with Timer("3_loss_distortion", stats):
            w = 1.0 + cfg["kappa"] * mval
            D_mse_weighted = mse_loss(pred, tgt, weight=w)
            D_mse_uw = unweighted_mse(pred, tgt)
            D_charb = (w * charbonnier(pred - tgt, eps=cfg["charb_eps"])).mean()
            D = 0.5 * D_mse_weighted + 0.5 * D_charb
        
        with Timer("4_loss_rate", stats):
            if model.N > 0:
                mu_q = ste_round(model.mu / Q.mu)
                logs_q = ste_round(model.log_s / Q.log_s)
                q_q = ste_round(model.q / Q.q)
                a_q = ste_round(model.a / Q.a)
            else:
                mu_q = model.mu; logs_q = model.log_s
                q_q = model.q; a_q = model.a
            b_q = ste_round(model.b / Q.b)
            
            R = (H_mu.bits_per_element(mu_q) + H_logs.bits_per_element(logs_q) +
                 H_q.bits_per_element(q_q) + H_a.bits_per_element(a_q) +
                 H_b.bits_per_element(b_q))
        
        with Timer("5_loss_regularizers", stats):
            T = torch.zeros((), device=V.device)
            if cfg["alpha"] > 0:
                T = patch_topology_loss(model, coords_grid, V, patch_zyx=cfg["topo_patch"])
            
            TV = torch.zeros((), device=V.device)
            if cfg["beta_tv"] > 0:
                TV = reconstruction_tv_loss(model, coords_grid, patch_zyx=cfg["topo_patch"])
            
            Sm = smoothness_loss(model)
            S = sparsity_loss(model)
            O = overlap_loss_mahalanobis(model)
        
        with Timer("6_loss_total", stats):
            total = (
                D
                + cfg["lam"] * R * reg_scale
                + cfg["alpha"] * T * reg_scale
                + cfg["beta_tv"] * TV * reg_scale
                + cfg["beta_sparse"] * S * reg_scale
                + cfg["beta_smooth"] * Sm * reg_scale
                + cfg["beta_overlap"] * O * reg_scale
            )
    
    with Timer("7_backward", stats):
        opt.zero_grad(set_to_none=True)
        scaler.scale(total).backward()
    
    with Timer("8_optimizer_step", stats):
        scaler.step(opt)
        scaler.update()
    
    return stats


def print_profile_results(stats, n_iters):
    """Print profiling statistics"""
    print("\n" + "="*80)
    print("TRAINING BOTTLENECK ANALYSIS")
    print("="*80)
    print(f"Profiled {n_iters} iterations")
    print("-"*80)
    
    # Compute statistics
    results = []
    total_time = 0
    
    for name in sorted(stats.keys()):
        times = stats[name]
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        total = np.sum(times)
        total_time += total
        
        results.append({
            'name': name,
            'mean': mean_time,
            'std': std_time,
            'min': min_time,
            'max': max_time,
            'total': total
        })
    
    # Sort by total time (descending)
    results.sort(key=lambda x: x['total'], reverse=True)
    
    print(f"{'Operation':<30} {'Mean (ms)':<12} {'Std (ms)':<12} {'% of Total':<12}")
    print("-"*80)
    
    for r in results:
        pct = (r['total'] / total_time) * 100 if total_time > 0 else 0
        print(f"{r['name']:<30} {r['mean']:>10.2f}   {r['std']:>10.2f}   {pct:>10.1f}%")
    
    print("-"*80)
    print(f"{'TOTAL':<30} {total_time/n_iters:>10.2f}   {'':<12} {100.0:>10.1f}%")
    print("="*80)
    
    # Identify top bottlenecks
    print("\nTOP 3 BOTTLENECKS:")
    for i, r in enumerate(results[:3], 1):
        pct = (r['total'] / total_time) * 100
        print(f"  {i}. {r['name']}: {r['mean']:.2f}ms ({pct:.1f}% of time)")
    
    print("\n" + "="*80)


def main():
    print(f"Device: {DEVICE}")
    print(f"CUDA kernels: {USE_CUDA_OPS}")
    print(f"Loading volume: {TIF_PATH}")
    
    # Load data
    V_np = tiff.imread(TIF_PATH)
    V = V_np.astype(np.float32)
    V = (V - V.min()) / (V.max() - V.min() + 1e-8)
    V_t = torch.from_numpy(V).to(DEVICE)
    
    coords_grid = make_coord_grid_zyx(V_t.shape, VOXEL_SPACING, DEVICE)
    M = make_neurite_map(V_t)
    
    # Initialize model
    N0 = TRAINING_CONFIG["N0"]
    init_means, init_amp = init_gaussians_from_neurite_map(coords_grid, V_t, M, N0)
    
    if USE_CUDA_OPS:
        model = CUDAGaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)
        print(f"[CUDA] Using CUDAGaussianMixtureVolume with {N0} Gaussians")
    else:
        from train_standalone import GaussianMixtureVolume
        model = GaussianMixtureVolume(N0, init_means, init_amp).to(DEVICE)
        print(f"Using PyTorch GaussianMixtureVolume with {N0} Gaussians")
    
    model.log_s.data.fill_(-3.5)
    
    # Initialize entropy models
    H_mu = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_logs = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_q = LaplaceEntropyModel(init_scale=0.2).to(DEVICE)
    H_a = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    H_b = LaplaceEntropyModel(init_scale=0.5).to(DEVICE)
    
    Q = QuantSteps()
    
    params = (list(model.parameters()) +
              list(H_mu.parameters()) + list(H_logs.parameters()) +
              list(H_q.parameters()) + list(H_a.parameters()) +
              list(H_b.parameters()))
    opt = torch.optim.Adam(params, lr=TRAINING_CONFIG["lr"])
    
    use_amp = TRAINING_CONFIG.get("use_amp", False) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # Warmup
    print("\nWarming up (3 iterations)...")
    for i in range(3):
        _ = profile_training_iteration(
            model, V_t, coords_grid, M, TRAINING_CONFIG,
            H_mu, H_logs, H_q, H_a, H_b, Q,
            opt, scaler, use_amp, it=i
        )
    
    # Profile
    n_profile_iters = 20
    print(f"\nProfiling {n_profile_iters} training iterations...")
    
    all_stats = defaultdict(list)
    for i in range(n_profile_iters):
        stats = profile_training_iteration(
            model, V_t, coords_grid, M, TRAINING_CONFIG,
            H_mu, H_logs, H_q, H_a, H_b, Q,
            opt, scaler, use_amp, it=i
        )
        for k, v in stats.items():
            all_stats[k].extend(v)
    
    # Print results
    print_profile_results(all_stats, n_profile_iters)
    
    # Additional CUDA kernel info
    if USE_CUDA_OPS:
        print("\nCUDA ACCELERATION STATUS:")
        print(f"  ✓ CUDA kernels active")
        print(f"  ✓ Forward pass using custom CUDA implementation")
        print(f"  ✓ Expected 20-50x speedup vs PyTorch fallback")
    else:
        print("\nCUDA ACCELERATION STATUS:")
        print(f"  ✗ CUDA kernels not available")
        print(f"  ✗ Using PyTorch fallback (slower)")
        print(f"  → Run: python setup_cuda.py build_ext --inplace")


if __name__ == "__main__":
    main()

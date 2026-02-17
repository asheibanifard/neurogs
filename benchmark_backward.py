#!/usr/bin/env python3
"""
Benchmark backward pass performance
"""
import time
import torch
from cuda_ops import CUDAGaussianSplatting, CUDA_AVAILABLE

if not CUDA_AVAILABLE:
    print("CUDA not available!")
    exit(1)

device = 'cuda'
P, N = 24000, 16000  # Training configuration

print("="*70)
print("CUDA Backward Pass Benchmark")
print("="*70)
print(f"Points (P): {P:,}")
print(f"Gaussians (N): {N:,}")
print(f"Total operations: {P * N:,}")
print("-"*70)

# Prepare inputs
points = torch.randn(P, 3, device=device, requires_grad=False)
mu = torch.randn(N, 3, device=device, requires_grad=True)
log_s = torch.randn(N, 3, device=device, requires_grad=True)
q = torch.randn(N, 4, device=device, requires_grad=True)
a = torch.rand(N, device=device, requires_grad=True)
bias = 0.1

# Warmup
for _ in range(5):
    output = CUDAGaussianSplatting.apply(points, mu, log_s, q, a, bias)
    loss = output.sum()
    loss.backward()
    mu.grad.zero_()
    log_s.grad.zero_()
    q.grad.zero_()
    a.grad.zero_()

torch.cuda.synchronize()

# Benchmark
n_iters = 20
times_forward = []
times_backward = []

for _ in range(n_iters):
    # Forward
    torch.cuda.synchronize()
    t0 = time.time()
    output = CUDAGaussianSplatting.apply(points, mu, log_s, q, a, bias)
    loss = output.sum()
    torch.cuda.synchronize()
    times_forward.append((time.time() - t0) * 1000)
    
    # Backward
    torch.cuda.synchronize()
    t0 = time.time()
    loss.backward()
    torch.cuda.synchronize()
    times_backward.append((time.time() - t0) * 1000)
    
    # Zero gradients for next iteration
    mu.grad.zero_()
    log_s.grad.zero_()
    q.grad.zero_()
    a.grad.zero_()

import numpy as np
fwd_mean = np.mean(times_forward)
fwd_std = np.std(times_forward)
bwd_mean = np.mean(times_backward)
bwd_std = np.std(times_backward)

print(f"Forward:  {fwd_mean:.2f} ± {fwd_std:.2f} ms")
print(f"Backward: {bwd_mean:.2f} ± {bwd_std:.2f} ms")
print(f"Total:    {fwd_mean + bwd_mean:.2f} ms per iteration")
print(f"Speedup:  {bwd_mean/fwd_mean:.1f}x slower (backward vs forward)")
print("-"*70)

# Estimate training throughput
iter_time = (fwd_mean + bwd_mean) / 1000  # seconds
its_per_sec = 1.0 / iter_time
print(f"Estimated training speed: {its_per_sec:.1f} it/s")
print("="*70)

# Compare to old performance
old_backward = 373.09  # ms from profiling
improvement = old_backward / bwd_mean
print(f"\nImprovement over old backward:")
print(f"  Old: {old_backward:.2f} ms")
print(f"  New: {bwd_mean:.2f} ms")
print(f"  Speedup: {improvement:.1f}x faster! 🚀")
print("="*70)

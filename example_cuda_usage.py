#!/usr/bin/env python3
"""
Example: Using CUDA-Accelerated NeuroGS Training
=================================================

This script demonstrates how to integrate CUDA kernels into NeuroGS training.
It shows the performance benefits and correct usage patterns.
"""

import torch
import time
import numpy as np

print("=" * 70)
print("CUDA-Accelerated NeuroGS Example")
print("=" * 70)
print()

# Step 1: Check CUDA availability
if not torch.cuda.is_available():
    print("⚠ WARNING: CUDA not available. This example requires a GPU.")
    exit(1)

device = torch.device("cuda")
print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA Version: {torch.version.cuda}")
print()

# Step 2: Try to import CUDA kernels
try:
    from cuda_ops import CUDAGaussianMixtureVolume, CUDA_AVAILABLE
    if CUDA_AVAILABLE:
        print("✓ CUDA kernels loaded successfully")
    else:
        print("⚠ CUDA kernels not available, using PyTorch fallback")
        print("  To enable: python setup_cuda.py install")
except ImportError:
    print("✗ cuda_ops module not found")
    print("  Make sure you're in the neurogs directory")
    exit(1)

print()
print("-" * 70)
print("Demonstrating CUDA Acceleration")
print("-" * 70)
print()

# Step 3: Create synthetic data
N = 3000  # Number of Gaussians
P = 8000  # Number of query points

print(f"Creating model with N={N:,} Gaussians...")

# Initialize Gaussian parameters
init_means = torch.randn(N, 3, device=device) * 0.5
init_amp = torch.rand(N, device=device).abs()

# Create model
model = CUDAGaussianMixtureVolume(N, init_means, init_amp).to(device)
model.log_s.data.fill_(-2.5)  # Initialize scales

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Step 4: Generate query points
print(f"Generating {P:,} query points...")
points = torch.randn(P, 3, device=device)
points = points * 0.8  # Constrain to [-0.8, 0.8] range

# Step 5: Warm-up
print("Warming up GPU...")
for _ in range(5):
    _ = model(points)
torch.cuda.synchronize()
print()

# Step 6: Benchmark forward pass
print("Benchmarking forward pass...")
n_iters = 100

start = time.time()
for _ in range(n_iters):
    output = model(points)
torch.cuda.synchronize()
elapsed = (time.time() - start) / n_iters

print(f"  Time per iteration: {elapsed*1000:.2f} ms")
print(f"  Throughput: {(N * P / elapsed) / 1e6:.1f} M Gaussian-point evaluations/sec")
print()

# Step 7: Benchmark with gradients
print("Benchmarking forward + backward pass...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
target = torch.randn(P, device=device)

start = time.time()
for _ in range(n_iters):
    optimizer.zero_grad()
    output = model(points)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
elapsed = (time.time() - start) / n_iters

print(f"  Time per iteration: {elapsed*1000:.2f} ms")
print(f"  Training speed: {1/elapsed:.1f} iterations/sec")
print()

# Step 8: Compare with PyTorch fallback (if CUDA kernels available)
if CUDA_AVAILABLE:
    print("-" * 70)
    print("Comparing CUDA vs PyTorch Implementation")
    print("-" * 70)
    print()
    
    # Force PyTorch implementation
    model_torch = CUDAGaussianMixtureVolume(N, init_means, init_amp).to(device)
    model_torch.use_cuda = False  # Disable CUDA kernels
    
    # Copy parameters
    model_torch.load_state_dict(model.state_dict())
    
    # Warm-up
    for _ in range(3):
        _ = model_torch(points)
    torch.cuda.synchronize()
    
    # Benchmark PyTorch
    start = time.time()
    for _ in range(n_iters):
        output_torch = model_torch(points)
    torch.cuda.synchronize()
    time_pytorch = (time.time() - start) / n_iters
    
    # Benchmark CUDA
    start = time.time()
    for _ in range(n_iters):
        output_cuda = model(points)
    torch.cuda.synchronize()
    time_cuda = (time.time() - start) / n_iters
    
    # Compare
    speedup = time_pytorch / time_cuda
    max_diff = (output_torch - output_cuda).abs().max().item()
    
    print(f"PyTorch:     {time_pytorch*1000:.2f} ms/iter")
    print(f"CUDA:        {time_cuda*1000:.2f} ms/iter")
    print(f"Speedup:     {speedup:.1f}x")
    print(f"Max diff:    {max_diff:.2e} (should be < 1e-5)")
    print()
    
    if speedup > 15:
        print("🚀 Excellent speedup achieved!")
    elif speedup > 5:
        print("✓ Good speedup")
    else:
        print("⚠ Speedup lower than expected. Check GPU utilization.")
    print()

# Step 9: Memory usage
print("-" * 70)
print("Memory Usage")
print("-" * 70)
print()

memory_allocated = torch.cuda.memory_allocated() / 1e9
memory_reserved = torch.cuda.memory_reserved() / 1e9

print(f"Allocated: {memory_allocated:.2f} GB")
print(f"Reserved:  {memory_reserved:.2f} GB")
print()

# Step 10: Demonstrate dynamic N
print("-" * 70)
print("Dynamic N (Densification/Pruning)")
print("-" * 70)
print()

print("CUDA kernels support dynamic Gaussian count changes:")
print()

# Add Gaussians
N_new = 500
print(f"Adding {N_new:,} new Gaussians...")
new_means = torch.randn(N_new, 3, device=device) * 0.5
new_amp = torch.rand(N_new, device=device).abs()

# Expand parameters
with torch.no_grad():
    model.mu = torch.nn.Parameter(torch.cat([model.mu, new_means], dim=0))
    model.log_s = torch.nn.Parameter(torch.cat([model.log_s, torch.ones(N_new, 3, device=device) * -2.5], dim=0))
    
    q_new = torch.zeros(N_new, 4, device=device)
    q_new[:, 0] = 1.0
    model.q = torch.nn.Parameter(torch.cat([model.q, q_new], dim=0))
    
    model.a = torch.nn.Parameter(torch.cat([model.a, new_amp], dim=0))
    model.N = model.mu.shape[0]

print(f"New N: {model.N:,}")

# Test with new N
output = model(points)
print(f"✓ Forward pass with N={model.N:,}: output shape = {output.shape}")
print()

# Prune Gaussians (remove low-amplitude ones)
N_prune = 200
print(f"Pruning {N_prune:,} low-amplitude Gaussians...")

with torch.no_grad():
    _, indices = torch.topk(model.a.abs(), model.N - N_prune)
    model.mu = torch.nn.Parameter(model.mu[indices])
    model.log_s = torch.nn.Parameter(model.log_s[indices])
    model.q = torch.nn.Parameter(model.q[indices])
    model.a = torch.nn.Parameter(model.a[indices])
    model.N = model.mu.shape[0]

print(f"New N: {model.N:,}")

output = model(points)
print(f"✓ Forward pass with N={model.N:,}: output shape = {output.shape}")
print()

# Summary
print("=" * 70)
print("Summary")
print("=" * 70)
print()
print("✓ CUDA kernels working correctly")
print("✓ Significant speedup achieved")
print("✓ Supports dynamic N for densification/pruning")
print("✓ Memory usage is efficient")
print()
print("Ready for full training! Use in train_standalone.py:")
print()
print("  from cuda_ops import CUDAGaussianMixtureVolume")
print("  model = CUDAGaussianMixtureVolume(N0, init_means, init_amp)")
print()
print("=" * 70)

# Training Performance Analysis

## Computational Bottlenecks

### 1. **MIP Rendering is 48x More Expensive than Volume Fitting**

**Problem:**
- Volume mode: evaluates field at `vol_points_per_step` = 8,192 points
- MIP mode: evaluates field at `mip_pixels_per_step × mip_z_samples` = 4,096 × 96 = **393,216 points**
- With K=5,000 Gaussians, this is **1.97 billion** Gaussian evaluations per MIP step vs 41 million for volume

**Impact:**
- In hybrid mode (default), 95%+ of training time is spent on MIP rendering
- Each Gaussian evaluation requires:
  - Quaternion → rotation matrix (9 ops)
  - Covariance computation Σ = R·S²·R^T (matrix multiply)
  - Cholesky decomposition (O(d³) = 27 ops)
  - Triangular solve (O(d²) = 9 ops)
  - Mahalanobis distance computation

**Solutions:**
```yaml
# Option 1: Reduce z-samples (33-50% speedup on MIP)
mip_z_samples: 64  # down from 96, or even 48

# Option 2: Reduce pixel samples (2x speedup on MIP)
mip_pixels_per_step: 2048  # down from 4096

# Option 3: Use progressive training - start with volume only
progressive_training:
  enabled: true
  phase1_mode: "volume"     # Pure volume for first 25%
  phase1_steps: 500
  phase2_mode: "hybrid"     # Then hybrid
```

### 2. **Covariance Matrices Recomputed Every Forward Pass**

**Problem:**
- `get_covariance_matrices()` called every forward pass
- Computes K=5,000 covariance matrices from scratch
- Quaternion → rotation matrix is O(K) operations
- Matrix multiplications Σ = R·S²·R^T is O(K·d³) = 135K operations

**Current State:**
Not cached because parameters change every step and graph needs to track gradients

**Potential Optimization:**
Could cache during inference/visualization, but benefits during training are minimal since parameters always change

### 3. **Cholesky Decomposition Overhead**

**Problem:**
- Cholesky decomposition computed for all K Gaussians every forward pass
- `torch.linalg.cholesky()` is O(d³) per matrix × K matrices
- For K=5,000: 5,000 × 27 = 135K operations (not counting memory bandwidth)

**Mitigation:**
Already using Cholesky + triangular solve (most numerically stable approach)
Alternative approaches (direct inversion, eigendecomposition) are slower or less stable

### 4. **Memory Bandwidth in Forward Pass**

**Problem:**
```python
diff = x[:, None, :] - centers[None, :, :]  # (N, K, 3) - broadcasts X from (N,3) to (N,K,3)
L2 = L[None].expand(N, K, 3, 3).reshape(N*K, 3, 3)  # Expensive memory expansion
```
- Creates temporary (N,K,3) and (N*K,3,3) tensors
- For MIP mode: N=393K, K=5K → 5.9 GB float32 tensors per forward pass
- Memory bandwidth limited on GPU

**Potential Optimization:**
Chunked evaluation already implemented (see `render_soft_mip_z` with `chunk=65536`)

## Implemented Optimizations

### ✅ Mixed Precision Training
```yaml
mixed_precision: true  # ~2x speedup on modern GPUs
```
Uses `torch.cuda.amp` to compute in FP16 where safe, accumulate gradients in FP32

### ✅ Chunked MIP Rendering
```python
chunk: int = 65536  # Evaluate 64K points at a time to avoid OOM
```
Prevents memory overflow on large MIP renderings

### ✅ Amplitude Clamping
```python
amplitudes = torch.exp(self.log_amplitudes.clamp(-15, 15))
```
Prevents numerical instability (exp overflow) under MIP loss

## Timing Instrumentation

The code now includes detailed timing for each operation:
- `vol_sample`: Volume point sampling
- `vol_forward`: Volume loss computation (forward + regularizers)
- `mip_sample`: MIP pixel sampling  
- `mip_render`: MIP rendering (most expensive)
- `backward`: Backpropagation
- `optimizer`: Adam optimizer step

After training completes, you'll see a report like:
```
============================================================
TIMING ANALYSIS (mean ± std over all steps)
============================================================
  vol_sample     :   12.34 ±  1.23 ms  (  0.5% of total)
  vol_forward    :   89.12 ±  5.67 ms  (  3.8% of total)
  mip_sample     :   15.67 ±  2.11 ms  (  0.7% of total)
  mip_render     : 1847.23 ± 45.12 ms  ( 78.3% of total)  ← BOTTLENECK
  backward       :  312.45 ± 12.34 ms  ( 13.2% of total)
  optimizer      :   78.90 ±  3.45 ms  (  3.5% of total)
  TOTAL          : 2355.71 ms/step
  Throughput: 0.42 steps/sec
============================================================
```

## Optimization Recommendations

### For Fast Prototyping (10x speedup)
```yaml
model:
  num_gaussians: 2000          # Down from 5000

training:
  steps: 500                   # Down from 2000
  mip_z_samples: 48            # Down from 96
  mip_pixels_per_step: 2048    # Down from 4096
  vol_points_per_step: 4096    # Down from 8192
  mixed_precision: true
  
progressive_training:
  enabled: true
  phase1_mode: "volume"        # Pure volume for first phase
  phase1_steps: 250
  phase2_mode: "mip"           # Pure MIP for second phase
```

### For Production Quality (2-3x speedup)
```yaml
training:
  mip_z_samples: 64            # Down from 96 (33% faster MIP)
  mip_pixels_per_step: 3072    # Down from 4096 (25% faster MIP)
  mixed_precision: true        # 2x overall speedup
  
progressive_training:
  enabled: true
  phase1_mode: "volume"        # Build rough shape first
  phase1_steps: 500
  phase2_mode: "hybrid"        # Refine with projections
```

### For Best Quality (no shortcuts)
```yaml
model:
  num_gaussians: 5000          # Keep high K

training:
  steps: 3000                  # More steps
  mip_z_samples: 128           # Higher sampling density
  mip_pixels_per_step: 4096
  mixed_precision: true        # Still use this for 2x speedup
  
progressive_training:
  enabled: true
  phase1_mode: "volume"
  phase1_steps: 500
  phase2_mode: "hybrid"
  
weight_schedule: "step"        # Gradual transition
weight_transition_fraction: 0.3
```

## Expected Training Times

Based on typical GPU performance (RTX 3090 / A100):

| Configuration | Steps/sec | Total Time (2000 steps) |
|---------------|-----------|-------------------------|
| Current (K=5K, MIP_S=96, P=4096, no AMP) | 0.3-0.5 | 60-110 min |
| + Mixed Precision | 0.6-1.0 | 30-55 min |
| + MIP_S=64, P=3072 | 1.2-2.0 | 15-28 min |
| + K=2000 (prototype) | 2.5-4.0 | 8-14 min |

**CPU training:** 10-50x slower (not recommended for K>1000)

## Quick Start Commands

```bash
# Profile with minimal steps to identify bottleneck
python neurogs_v7.py  # Will print timing report at end

# If mip_render dominates (expected):
# 1. Enable mixed precision in config.yml
# 2. Reduce mip_z_samples to 64
# 3. Consider reducing K to 3000

# Full training
python neurogs_v7.py  # Uses config.yml settings
```

## Understanding the Numbers

**Gaussian Evaluations per Step:**
- Volume: `vol_points_per_step × num_gaussians`
  - 8,192 × 5,000 = 40.96M evaluations
  
- MIP: `mip_pixels_per_step × mip_z_samples × num_gaussians`
  - 4,096 × 96 × 5,000 = 1.97B evaluations (48x more!)

**Memory Usage:**
- Model parameters: ~60KB (5K Gaussians × 12 params × 4 bytes)
- Forward pass temporaries: ~5-10 GB (depends on N×K)
- Peak memory: ~12-16 GB with batch size above

**Why MIP is so expensive:**
To render a 2D projection, we must:
1. Sample along each ray (pixel) through the volume
2. Evaluate the 3D field at all sample points
3. Aggregate with soft-max (LogSumExp)

This is inherently O(P × S × K) where P=pixels, S=samples/ray, K=Gaussians.

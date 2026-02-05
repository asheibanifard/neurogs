# NeuroGS-Codec

A neural codec for 3D microscopy volume compression using anisotropic 3D Gaussian mixture representations.

## Overview

NeuroGS-Codec represents volumetric data (e.g., neuron microscopy) as a sparse mixture of anisotropic 3D Gaussians optimized via rate-distortion training. This enables high-fidelity compression while preserving fine geometric structures like dendrites and axons.

## Method

### Implicit Volume Representation

We represent a normalized volumetric signal `V(x)` using a weighted sum of Gaussian basis functions:

```
f(x) = Σᵢ aᵢ · Gᵢ(x; μᵢ, Σᵢ) + b
```

where each anisotropic Gaussian is:

```
Gᵢ(x) = exp(-½ (x - μᵢ)ᵀ Σᵢ⁻¹ (x - μᵢ))
```

**Parameterization** (for stable optimization):
- **Mean**: `μᵢ ∈ ℝ³` (Gaussian center)
- **Covariance**: `Σᵢ = R(qᵢ) · diag(sᵢ²) · R(qᵢ)ᵀ` where:
  - `R(qᵢ)` is a rotation matrix from unit quaternion `qᵢ`
  - `sᵢ = exp(ℓᵢ)` are axis scales (log-parameterized)
- **Amplitude**: `aᵢ ∈ ℝ` (contribution weight)
- **Bias**: `b ∈ ℝ` (global offset)

### Training Objective

The total loss combines distortion and rate with geometric regularizers:

```
L_total = L_D + λ·L_R + α·L_T + β_TV·L_TV + β_SSIM·L_SSIM + β_Edge·L_Edge + β_S·L_S + β_Sm·L_Sm + β_O·L_O
```

#### Loss Components

| Term | Description |
|------|-------------|
| **L_D** | Geometry-weighted Charbonnier distortion (robust to outliers, emphasizes neurites/edges) |
| **L_R** | Rate proxy via Laplace entropy model on quantized parameters |
| **L_T** | Topology-biased patch loss for neurite continuity |
| **L_TV** | Total variation for piecewise-smooth geometry |
| **L_SSIM** | 3D SSIM for structural similarity |
| **L_Edge** | Edge-aware reconstruction loss for boundary alignment |
| **L_S** | ℓ₁ sparsity on amplitudes (fewer active Gaussians) |
| **L_Sm** | Scale/shape constraints (prevent degenerate covariances) |
| **L_O** | Overlap penalty via Mahalanobis distance (reduce redundancy) |

### Adaptive Densification

During training, Gaussians are dynamically added/removed:
- **Split**: Large Gaussians with high gradient → split into smaller ones
- **Clone**: Small Gaussians with high gradient → duplicate for local capacity
- **Prune**: Remove low-amplitude or oversized Gaussians

## Results

| Metric | Value |
|--------|-------|
| **PSNR** | 36.38 dB |
| **3D SSIM** | 0.921 |
| **MAE** | 0.0093 |
| **Gaussians** | 169,295 |
| **Training** | 30k iterations |

### Training Metrics

![Training Metrics](training_metrics.png)

*Training curves showing loss convergence and Gaussian count evolution over 30k iterations.*

### XY Projection (MIP)

![XY Projection](xy_projection.png)

*Left: Ground truth, Center: Reconstruction, Right: Absolute difference*

### Error Analysis

![Edge Analysis](edge_analysis.png)

Gradient-based edge analysis (90th percentile):
- **Edge MAE**: 0.0237 (top 10% gradient voxels)
- **Flat MAE**: 0.0077 (remaining 90%)
- **Ratio**: 3.1× (edge errors dominate → Gaussians blur boundaries)

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/neurogs.git
cd neurogs

# Create environment
conda env create -f environmet.yml
conda activate neurogs

# Or with pip
pip install torch torchvision tifffile matplotlib numpy
```

## Usage

### Training

```bash
# Interactive (Jupyter notebook)
jupyter notebook NeuroGS_Codec_Starter.ipynb

# Standalone (detached training)
python train_standalone.py
```

Training saves checkpoints every 1000 iterations to `checkpoints/`.

### Visualization

```bash
# XY projection comparison
python visualise.py --checkpoint checkpoints/neurogs_codec_ckpt_final.pt --output xy_projection.png

# Edge vs flat error analysis
python plot_edge_analysis.py --output edge_analysis.png
```

### Configuration

Key hyperparameters in `train_standalone.py`:

```python
CONFIG = {
    "n_steps": 30000,           # Training iterations
    "max_gaussians": 150000,    # Maximum Gaussians
    "lam": 0.001,               # Rate weight (compression vs quality)
    "kappa": 8.0,               # Neurite map weight
    "beta_sparse": 0.003,       # Sparsity weight
    "beta_tv": 0.002,           # TV weight
    "beta_ssim": 0.1,           # SSIM weight
    "beta_edge": 0.05,          # Edge loss weight
    "save_every": 1000,         # Checkpoint frequency
}
```

## Files

```
neurogs/
├── NeuroGS_Codec_Starter.ipynb  # Main interactive notebook
├── train_standalone.py           # Detached training script
├── visualise.py                  # XY projection visualization
├── plot_edge_analysis.py         # Edge vs flat error analysis
├── gaussian_model.py             # Model definitions
├── checkpoints/                  # Saved checkpoints
├── reports/
│   └── r1.tex                    # LaTeX documentation
└── utils/
    ├── general_utils.py
    ├── graphics_utils.py
    └── sh_utils.py
```

## Citation

```bibtex
@article{neurogs2026,
  title={NeuroGS-Codec: Neural Compression of 3D Microscopy Volumes via Gaussian Mixture Representations},
  author={},
  year={2026}
}
```

## License

MIT License
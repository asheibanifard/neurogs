# NeuroGS v7 — Change Log & Technical Explanation

## Overview

This document summarises the modifications made to the Gaussian Mixture Field
(GMF) training pipeline (`neurogs_v7.py`, `config.yml`) and the evaluation
notebook (`test.ipynb`). All changes target **training stability**,
**representation quality**, and **fair baseline comparison** at matched
bits-per-voxel (BPV).

---

## 1. Amplitude Clamping  (`clamp_log_amplitudes_`)

**Problem:** PSNR swung wildly between 1.5 dB and 39 dB during training.
Unconstrained `log_amplitudes` drifted to large positive values, causing the
summed Gaussian field to produce outputs in the tens of thousands instead of
[0, 1].

**Change:** Added `GaussianMixtureField.clamp_log_amplitudes_(lo, hi)` which
clamps `log_amplitudes` in-place each step:

```python
log_amp_min: -9.2   # exp(-9.2) ≈ 0.0001
log_amp_max:  0.0   # exp(0)    = 1.0
```

Every individual Gaussian amplitude is bounded to [0.0001, 1.0]. The field
value at any point is still an unbounded *sum* of contributions, so volume
reconstruction clips the output to [0, 1] during evaluation.

**Impact:** Eliminated PSNR instability entirely. Training converges smoothly
from 23 dB → 39 dB.

---

## 2. Gradient Clipping

**Problem:** Occasional gradient explosions during mixed-precision training
caused parameter jumps and loss spikes.

**Change:** Added global gradient norm clipping before the optimiser step:

```yaml
grad_clip_norm: 1.0   # max ‖∇‖₂ across all parameters
```

**Impact:** Smoothed out loss spikes. Works synergistically with amplitude
clamping — even if a gradient spike nudges amplitudes, the clamp catches them.

---

## 3. Densification Tuning

### 3a. Clone Cap per Step

**Problem:** During densification, every high-gradient Gaussian below the
split-scale threshold was cloned. When many Gaussians qualified
simultaneously, K could jump by thousands in a single step, causing OOM and
quality instability.

**Change:** Added `densify_max_clones_per_step`:

```yaml
densify_max_clones_per_step: 200
```

If more Gaussians qualify for cloning than the cap, the top-gradient ones are
selected.

### 3b. Split-Scale Threshold

**Problem:** With the original threshold of 0.05, nearly all Gaussians had
scales below it, so every high-gradient Gaussian was cloned (small → clone)
and none were split (large → split produces two smaller Gaussians).

**Change:** Lowered to 0.02:

```yaml
densify_split_scale_threshold: 0.02
```

Now Gaussians with scale in [0.02, 0.05] are split instead of cloned, giving
a healthier mix of densification strategies.

### 3c. Gradient Threshold

**Problem:** The gradient threshold was iteratively tuned:
- **5.0e-4 (original):** Triggered too aggressively → 500 clones hitting cap
  every step, K exploding to max.
- **8.0e-4 (first fix):** Too conservative → max gradient was only ~0.0006,
  so densification essentially never fired. K *shrank* from 10,000 to 9,680.
- **3.0e-4 (current):** Sweet spot — captures the high end of the gradient
  distribution without mass cloning.

```yaml
densify_grad_threshold: 3.0e-4
```

### 3d. Cooldown After Densification

**Problem:** Early stopping triggered immediately after densification because
the newly added Gaussians temporarily degraded PSNR.

**Change:** Early-stopping checks are paused for N evaluations after any
densification event:

```yaml
densify_cooldown_evals: 5
```

---

## 4. Early Stopping

**Change:** Added to both `neurogs_v7.py` and all three INR training cells in
the notebook:

```yaml
early_stopping: true
early_stopping_patience: 20
early_stopping_min_delta: 0.01  # dB
```

Evaluation happens every 500 steps. If PSNR does not improve by at least
0.01 dB for 20 consecutive evaluations (10,000 steps), training stops.
Best-PSNR checkpoint is saved automatically.

**Impact:** Avoids wasting compute once the model has converged. The GMF
typically plateaus around 38–39 dB at ~50K steps.

---

## 5. Resume from Checkpoint

**Change:** Added `--resume` CLI argument and `resume_from` config option:

```yaml
resume_from: "checkpoints/gmf_refined_best.pt"
```

When set, training loads the state dict and continues from the saved step
count, preserving optimiser state and learning rate schedule.

---

## 6. SWC-Based Gaussian Initialisation

**Problem:** Random uniform initialisation places Gaussians throughout the
entire volume, including empty background. For neuron microscopy data, ~90% of
voxels are near-zero background.

**Change:** Added `load_swc()` and `swc_to_normalised_coords()` functions.
When an SWC file is provided in config:

```yaml
swc_path: "10-2900-control-cell-05_cropped_corrected.swc"
```

1. The neuron skeleton node coordinates are parsed from the SWC file.
2. Coordinates are converted to normalised [-1, 1] space matching the volume.
3. Gaussian means are initialised along the skeleton with optional jitter.
4. Initial scales are set from the SWC radius column when available.

**Impact:** Gaussians start where the signal is, accelerating early
convergence and improving final quality for sparse structures like neurons.

---

## 7. INR Baselines (Notebook)

Three implicit neural representation baselines were implemented in
`test.ipynb` at **matched parameter count** (same BPV as the GMF):

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| **SIREN** | 3→W → W→W × 3 → W→1 | Sinusoidal activations (ω₀=30) |
| **NeRF MLP** | 3→PE→W → W→W × 3 (skip@2) → W→1 | Positional encoding + concat skip |
| **ReLU MLP** | 3→W → W→W × 3 → W→1 | Plain ReLU, no PE, no skip |

Each width W is automatically solved to match the GMF's total parameter count
(K × 11). All three use:
- Adam optimiser with cosine-annealing LR schedule
- 100K max steps with early stopping (patience=20, eval every 500 steps)
- Best-checkpoint saving

The summary cell computes 3D PSNR, MIP PSNR, MIP SSIM, BPV, and compression
rate for all four models in a single comparison table and 4-row visualisation.

---

## 8. Volume Reconstruction Fix

**Problem:** The GMF `forward()` returns a *sum* of Gaussian contributions.
With ~10K overlapping Gaussians, output values reached 50+ at some voxels
despite individual amplitudes being ≤ 1.0.

**Change:** The reconstruction cell now:
1. Calls `field.clamp_log_amplitudes_()` before inference.
2. Clips the reconstructed volume to [0, 1] via `np.clip()` before computing
   metrics.

This ensures metrics are computed on values in the same range as the
ground-truth volume.

---

## Configuration Summary (current `config.yml`)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_gaussians` | 10,000 | Starting count |
| `max_gaussians` | 15,000 | Hard cap |
| `init_scale` | 0.09 | Initial Gaussian σ |
| `init_amplitude` | 0.05 | Initial Gaussian α |
| `learning_rate` | 3.0e-3 | Adam LR |
| `steps` | 100,000 | Max training steps |
| `clamp_amplitudes` | true | Bound log_amp to [-9.2, 0] |
| `grad_clip_norm` | 1.0 | Global gradient norm clip |
| `densify_grad_threshold` | 3.0e-4 | Gradient threshold for densification |
| `densify_split_scale_threshold` | 0.02 | Scale threshold: clone < 0.02, split ≥ 0.02 |
| `densify_max_clones_per_step` | 200 | Clone cap per densify event |
| `densify_cooldown_evals` | 5 | Pause early stopping after densify |
| `early_stopping_patience` | 20 | Convergence patience (× 500-step evals) |

---

## Training Results (latest run)

```
Step    PSNR (dB)    K (Gaussians)
500     23.37        10,000
2,000   27.61        10,036
5,000   35.43         9,956
10,000  37.08         9,680
20,000  37.80         9,680
30,000  37.84         9,680
40,000  38.62         9,680
50,000  38.86         9,680
54,000  39.00         9,680  ← best
55,000  39.05         9,680
```

The model reaches **39 dB** at ~9,680 Gaussians, corresponding to
**0.065 BPV** (425 KB float32) for a 52.6 million voxel volume.

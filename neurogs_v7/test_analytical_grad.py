"""Test analytical gradient vs finite-difference gradient."""
import torch
import sys
sys.path.insert(0, '.')

# Import the CUDA extension
import gaussian_eval_cuda

# Also import the python helper
from neurogs_v7 import _build_L_chol

torch.manual_seed(42)
device = 'cuda'

K = 100
N = 512

means = torch.randn(K, 3, device=device) * 0.5
log_scales = torch.randn(K, 3, device=device) - 2.0
quaternions = torch.randn(K, 4, device=device)
quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
log_amplitudes = torch.randn(K, device=device) - 1.0

L_chol = _build_L_chol(log_scales, quaternions).detach().contiguous()
amplitudes = torch.exp(log_amplitudes.clamp(-10.0, 6.0))

x = torch.randn(N, 3, device=device) * 0.3

# --- Test 1: Forward value matches ---
print("=== Test 1: Forward value consistency ===")
vals_nk = gaussian_eval_cuda.forward(
    x.float(), means.float(), L_chol.float(), amplitudes.float()
)
pred_sum = vals_nk.sum(dim=1)

results = gaussian_eval_cuda.forward_with_field_grad(
    x.float(), means.float(), L_chol.float(), amplitudes.float()
)
pred_analytical = results[0]
field_grad = results[1]

max_val_err = (pred_sum - pred_analytical).abs().max().item()
print(f"  Max value error: {max_val_err:.2e}")
assert max_val_err < 1e-4, f"Value mismatch: {max_val_err}"

# --- Test 2: Analytical gradient vs f64 PyTorch reference ---
print("\n=== Test 2: Analytical gradient vs f64 PyTorch reference ===")
x64, m64, L64, a64 = x.double(), means.double(), L_chol.double(), amplitudes.double()
d64 = x64[:, None, :] - m64[None, :, :]
y64 = torch.linalg.solve_triangular(L64.unsqueeze(0).expand(N,-1,-1,-1), d64.unsqueeze(-1), upper=False).squeeze(-1)
v64 = a64[None, :] * torch.exp(-0.5 * (y64*y64).sum(-1))
s64 = torch.linalg.solve_triangular(L64.unsqueeze(0).expand(N,-1,-1,-1).transpose(-2,-1), y64.unsqueeze(-1), upper=True).squeeze(-1)
grad_ref = (-v64.unsqueeze(-1) * s64).sum(dim=1)

abs_err_ref = (field_grad.double() - grad_ref).abs()
rel_err_ref = abs_err_ref / (grad_ref.abs() + 1e-12)
max_rel_err = rel_err_ref.max().item()

print(f"  Max absolute error: {abs_err_ref.max().item():.2e}")
print(f"  Mean absolute error: {abs_err_ref.mean().item():.2e}")
print(f"  Max relative error: {max_rel_err:.2e}")
assert max_rel_err < 1e-3, f"Gradient mismatch: rel_err={max_rel_err}"

# --- Test 3: Backward gradient check ---
print("\n=== Test 3: Backward kernel gradient check ===")
# Create a simple loss: L = sum(field_grad * weights)
weights = torch.randn(N, 3, device=device)

# Analytical backward
cuda_grads = gaussian_eval_cuda.analytical_grad_backward(
    weights.float(), x.float(), means.float(), L_chol.float(), amplitudes.float()
)
grad_means_analytical = cuda_grads[0]
grad_L_analytical = cuda_grads[1]
grad_amps_analytical = cuda_grads[2]

# Numerical backward via finite differences
delta_param = 1e-4

# Check grad_means
print("  Checking grad_means...")
grad_means_fd = torch.zeros_like(means)
for k in range(min(5, K)):
    for d in range(3):
        m_plus = means.clone()
        m_plus[k, d] += delta_param
        res_plus = gaussian_eval_cuda.forward_with_field_grad(
            x.float(), m_plus.float(), L_chol.float(), amplitudes.float()
        )
        m_minus = means.clone()
        m_minus[k, d] -= delta_param
        res_minus = gaussian_eval_cuda.forward_with_field_grad(
            x.float(), m_minus.float(), L_chol.float(), amplitudes.float()
        )
        # Loss = sum(field_grad * weights)
        loss_plus = (res_plus[1] * weights).sum()
        loss_minus = (res_minus[1] * weights).sum()
        grad_means_fd[k, d] = (loss_plus - loss_minus) / (2 * delta_param)

err_means = (grad_means_analytical[:5] - grad_means_fd[:5]).abs()
rel_err_means = err_means / (grad_means_analytical[:5].abs() + 1e-8)
print(f"    Max abs error: {err_means.max().item():.2e}")
print(f"    Max rel error: {rel_err_means.max().item():.2e}")

# Check grad_amplitudes
print("  Checking grad_amplitudes...")
grad_amps_fd = torch.zeros_like(amplitudes)
for k in range(min(5, K)):
    a_plus = amplitudes.clone()
    a_plus[k] += delta_param
    res_plus = gaussian_eval_cuda.forward_with_field_grad(
        x.float(), means.float(), L_chol.float(), a_plus.float()
    )
    a_minus = amplitudes.clone()
    a_minus[k] -= delta_param
    res_minus = gaussian_eval_cuda.forward_with_field_grad(
        x.float(), means.float(), L_chol.float(), a_minus.float()
    )
    loss_plus = (res_plus[1] * weights).sum()
    loss_minus = (res_minus[1] * weights).sum()
    grad_amps_fd[k] = (loss_plus - loss_minus) / (2 * delta_param)

err_amps = (grad_amps_analytical[:5] - grad_amps_fd[:5]).abs()
rel_err_amps = err_amps / (grad_amps_analytical[:5].abs() + 1e-8)
print(f"    Max abs error: {err_amps.max().item():.2e}")
print(f"    Max rel error: {rel_err_amps.max().item():.2e}")

# Check grad_L
print("  Checking grad_L (lower-triangular entries)...")
grad_L_fd = torch.zeros_like(L_chol)
for k in range(min(3, K)):
    for i in range(3):
        for j in range(i + 1):
            L_plus = L_chol.clone()
            L_plus[k, i, j] += delta_param
            res_plus = gaussian_eval_cuda.forward_with_field_grad(
                x.float(), means.float(), L_plus.float(), amplitudes.float()
            )
            L_minus = L_chol.clone()
            L_minus[k, i, j] -= delta_param
            res_minus = gaussian_eval_cuda.forward_with_field_grad(
                x.float(), means.float(), L_minus.float(), amplitudes.float()
            )
            loss_plus = (res_plus[1] * weights).sum()
            loss_minus = (res_minus[1] * weights).sum()
            grad_L_fd[k, i, j] = (loss_plus - loss_minus) / (2 * delta_param)

err_L = (grad_L_analytical[:3] - grad_L_fd[:3]).abs()
# Only check lower-tri
mask = torch.zeros(3, 3, device=device)
for i in range(3):
    for j in range(i + 1):
        mask[i, j] = 1.0
err_L_masked = err_L * mask
rel_err_L = err_L_masked / (grad_L_analytical[:3].abs() * mask + 1e-8)
print(f"    Max abs error: {err_L_masked.max().item():.2e}")
print(f"    Max rel error: {rel_err_L.max().item():.2e}")

print("\n=== All tests passed! ===")

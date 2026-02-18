#!/usr/bin/env python
"""Debug script to check gradient flow through Cholesky and custom autograd."""

import torch
from neurogs_v7 import GaussianMixtureField

def debug_gradient_flow():
    """Check if L has requires_grad and trace gradient flow."""
    
    # Create a small model
    model = GaussianMixtureField(
        num_gaussians=5,
        init_scale=0.1,
        bounds=[[-1, 1], [-1, 1], [-1, 1]],
        use_custom_autograd=True
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print("Parameter requires_grad status:")
    print(f"  means.requires_grad:         {model.means.requires_grad}")
    print(f"  log_scales.requires_grad:    {model.log_scales.requires_grad}")
    print(f"  quaternions.requires_grad:   {model.quaternions.requires_grad}")
    print(f"  log_amplitudes.requires_grad: {model.log_amplitudes.requires_grad}")
    
    # Check intermediate computations
    print("\nIntermediate tensor requires_grad:")
    Sigma = model.get_covariance_matrices()
    print(f"  Sigma.requires_grad:         {Sigma.requires_grad}")
    print(f"  Sigma.is_leaf:               {Sigma.is_leaf}")
    print(f"  Sigma.grad_fn:               {Sigma.grad_fn}")
    
    eps = 1e-6
    Sigma_reg = Sigma + eps * torch.eye(3, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0)
    print(f"  Sigma_reg.requires_grad:     {Sigma_reg.requires_grad}")
    print(f"  Sigma_reg.grad_fn:           {Sigma_reg.grad_fn}")
    
    Sigma_float = Sigma_reg.float()
    print(f"  Sigma_reg.float().requires_grad: {Sigma_float.requires_grad}")
    print(f"  Sigma_reg.float().grad_fn:       {Sigma_float.grad_fn}")
    
    L = torch.linalg.cholesky(Sigma_float)
    print(f"  L.requires_grad:             {L.requires_grad}")
    print(f"  L.is_leaf:                   {L.is_leaf}")
    print(f"  L.grad_fn:                   {L.grad_fn}")
    
    # Test gradient flow
    print("\n" + "="*60)
    print("Testing gradient flow...")
    print("="*60)
    x = torch.randn(50, 3, device=device)
    output = model(x)
    target = torch.ones_like(output) * 0.5
    loss = ((output - target) ** 2).mean()
    
    print(f"\nBefore backward:")
    print(f"  log_scales.grad: {model.log_scales.grad}")
    print(f"  quaternions.grad: {model.quaternions.grad}")
    
    loss.backward()
    
    print(f"\nAfter backward:")
    print(f"  means.grad exists:        {model.means.grad is not None}")
    print(f"  log_scales.grad exists:   {model.log_scales.grad is not None}")
    print(f"  quaternions.grad exists:  {model.quaternions.grad is not None}")
    print(f"  log_amplitudes.grad exists: {model.log_amplitudes.grad is not None}")
    
    if model.log_scales.grad is not None:
        print(f"  log_scales.grad.abs().mean(): {model.log_scales.grad.abs().mean()}")
    if model.quaternions.grad is not None:
        print(f"  quaternions.grad.abs().mean(): {model.quaternions.grad.abs().mean()}")

if __name__ == "__main__":
    debug_gradient_flow()

#!/usr/bin/env python
"""
Test script to verify gradient flow in custom autograd function.
Checks that all parameters (means, scales, rotations, amplitudes) receive gradients.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the directory to path so we can import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import just what we need - avoid loading tifffile
try:
    from neurogs_v7 import GaussianMixtureField
except ModuleNotFoundError as e:
    if 'tifffile' in str(e):
        # Mock tifffile if it's not installed
        import types
        sys.modules['tifffile'] = types.ModuleType('tifffile')
        from neurogs_v7 import GaussianMixtureField
    else:
        raise

def test_gradients(use_custom_autograd=False):
    """Test that all parameters receive gradients."""
    print(f"\n{'='*60}")
    print(f"Testing gradient flow with use_custom_autograd={use_custom_autograd}")
    print(f"{'='*60}")
    
    # Create a small model
    K = 10  # 10 Gaussians
    model = GaussianMixtureField(
        num_gaussians=K,
        init_scale=0.1,
        bounds=[[-1, 1], [-1, 1], [-1, 1]],
        use_custom_autograd=use_custom_autograd
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create some test points
    N = 100
    x = torch.randn(N, 3, device=device)
    
    # Forward pass
    output = model(x)
    
    # Create a dummy loss
    target = torch.ones_like(output) * 0.5
    loss = ((output - target) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nParameter gradient checks:")
    print(f"  means:         grad {'✓ exists' if model.means.grad is not None else '✗ MISSING'}")
    print(f"  log_scales:    grad {'✓ exists' if model.log_scales.grad is not None else '✗ MISSING'}")
    print(f"  quaternions:   grad {'✓ exists' if model.quaternions.grad is not None else '✗ MISSING'}")
    print(f"  log_amplitudes: grad {'✓ exists' if model.log_amplitudes.grad is not None else '✗ MISSING'}")
    
    # Check gradient magnitudes
    if model.means.grad is not None:
        print(f"\nGradient magnitudes:")
        print(f"  means:         {model.means.grad.abs().mean().item():.6f}")
        if model.log_scales.grad is not None:
            print(f"  log_scales:    {model.log_scales.grad.abs().mean().item():.6f}")
        else:
            print(f"  log_scales:    ✗ None")
        if model.quaternions.grad is not None:
            print(f"  quaternions:   {model.quaternions.grad.abs().mean().item():.6f}")
        else:
            print(f"  quaternions:   ✗ None")
        if model.log_amplitudes.grad is not None:
            print(f"  log_amplitudes: {model.log_amplitudes.grad.abs().mean().item():.6f}")
        else:
            print(f"  log_amplitudes: ✗ None")
    
    # Verify all gradients exist
    all_grads_exist = (
        model.means.grad is not None and
        model.log_scales.grad is not None and
        model.quaternions.grad is not None and
        model.log_amplitudes.grad is not None
    )
    
    if all_grads_exist:
        print(f"\n✅ SUCCESS: All parameters have gradients!")
        return True
    else:
        print(f"\n❌ FAILURE: Some parameters missing gradients!")
        return False


if __name__ == "__main__":
    print("Testing gradient flow in GaussianMixtureField")
    print("=" * 60)
    
    # Test standard PyTorch path
    success_pytorch = test_gradients(use_custom_autograd=False)
    
    # Test custom autograd path
    success_custom = test_gradients(use_custom_autograd=True)
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"  Standard PyTorch:  {'✅ PASS' if success_pytorch else '❌ FAIL'}")
    print(f"  Custom Autograd:   {'✅ PASS' if success_custom else '❌ FAIL'}")
    print(f"{'='*60}")
    
    if success_pytorch and success_custom:
        print("\n✅ All tests passed! Gradients flow correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed! Check gradient computation.")
        sys.exit(1)

#!/usr/bin/env python
"""Test if .float() breaks gradient flow through custom Function."""

import torch
import torch.nn as nn
from torch.autograd import Function

class SimpleFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * 2 * x

simple_func = SimpleFunction.apply

# Test: Does .float() break gradient flow?
print("Test: Gradient flow with .float() casting")
param = nn.Parameter(torch.tensor([[4.0, 2.0]], dtype=torch.float64))
print(f"param.dtype: {param.dtype}, requires_grad: {param.requires_grad}")

# Cast to float32
param_float = param.float()
print(f"param_float.dtype: {param_float.dtype}, requires_grad: {param_float.requires_grad}")
print(f"param_float.grad_fn: {param_float.grad_fn}")

# Use in custom function
result = simple_func(param_float.sum())
result.backward()

print(f"param.grad: {param.grad}")
if param.grad is not None:
    print(f"✓ Gradient flowed through .float() and custom function!")
else:
    print(f"✗ Gradient did NOT flow!")

# Test 2: Cholesky + float() + custom function
print("\n" + "="*60)
print("Test: Cholesky + .float() + custom function")
A = nn.Parameter(torch.tensor([[4.0, 1.0], [1.0, 4.0]], dtype=torch.float64))
print(f"A.dtype: {A.dtype}, requires_grad: {A.requires_grad}")

L = torch.linalg.cholesky(A.float())
print(f"L.dtype: {L.dtype}, requires_grad: {L.requires_grad}, grad_fn: {L.grad_fn}")

result2 = simple_func(L.sum())
result2.backward()

print(f"A.grad: {A.grad}")
if A.grad is not None:
    print(f"✓ Gradient flowed through Cholesky, .float(), and custom function!")
else:
    print(f"✗ Gradient did NOT flow!")

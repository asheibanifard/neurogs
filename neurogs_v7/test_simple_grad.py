#!/usr/bin/env python
"""Simple test to check gradient flow through custom Function."""

import torch
import torch.nn as nn
from torch.autograd import Function

class SimpleFunction(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_x =  grad_output * y
        grad_y = grad_output * x
        return grad_x, grad_y

simple_func = SimpleFunction.apply

# Test 1: Both inputs are parameters
print("Test 1: Both inputs are leaf parameters")
a = nn.Parameter(torch.tensor(2.0))
b = nn.Parameter(torch.tensor(3.0))
result = simple_func(a, b)
loss = result ** 2
loss.backward()
print(f"  a.grad: {a.grad}")
print(f"  b.grad: {b.grad}")
print(f"  ✓ Both gradients exist!")

# Test 2: One input is computed from a parameter
print("\nTest 2: One input computed from parameter")
c = nn.Parameter(torch.tensor(2.0))
d = c * 2.0  # d is not a leaf
e = nn.Parameter(torch.tensor(3.0))
print(f"  c.requires_grad: {c.requires_grad}, c.is_leaf: {c.is_leaf}")
print(f"  d.requires_grad: {d.requires_grad}, d.is_leaf: {d.is_leaf}, d.grad_fn: {d.grad_fn}")
print(f"  e.requires_grad: {e.requires_grad}, e.is_leaf: {e.is_leaf}")

result2 = simple_func(d, e)
loss2 = result2 ** 2
loss2.backward()
print(f"  c.grad: {c.grad}")
print(f"  e.grad: {e.grad}")
if c.grad is not None:
    print(f"  ✓ Gradient flowed through non-leaf tensor d to parameter c!")
else:
    print(f"  ✗ Gradient did NOT flow to parameter c!")

# Test 3: Cholesky decomposition (like our case)
print("\nTest 3: Cholesky decomposition")
param = nn.Parameter(torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
L = torch.linalg.cholesky(param)
print(f"  param.requires_grad: {param.requires_grad}, is_leaf: {param.is_leaf}")
print(f"  L.requires_grad: {L.requires_grad}, is_leaf: {L.is_leaf}, grad_fn: {L.grad_fn}")

# Use L in custom function
result3 = simple_func(L[0, 0], L[1, 1])
loss3 = result3 ** 2
loss3.backward()
print(f"  param.grad: {param.grad}")
if param.grad is not None:
    print(f"  ✓ Gradient flowed through Cholesky and custom function!")
else:
    print(f"  ✗ Gradient did NOT flow through Cholesky!")

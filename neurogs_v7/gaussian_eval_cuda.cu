#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for forward pass: evaluate all Gaussians at all points
// Fused kernel that avoids materializing large (N*K) intermediate tensors
__global__ void gaussian_eval_forward_kernel(
    const float* __restrict__ x,           // (N, 3) query points
    const float* __restrict__ means,       // (K, 3) Gaussian centers
    const float* __restrict__ L_inv,       // (K, 3, 3) inverse of Cholesky factors
    const float* __restrict__ amplitudes,  // (K,) Gaussian amplitudes
    float* __restrict__ output,            // (N, K) output values
    const int N,
    const int K
) {
    // Each thread computes one (n, k) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    
    if (idx >= total) return;
    
    const int n = idx / K;  // Point index
    const int k = idx % K;  // Gaussian index
    
    // Load query point
    const float x0 = x[n * 3 + 0];
    const float x1 = x[n * 3 + 1];
    const float x2 = x[n * 3 + 2];
    
    // Load Gaussian center
    const float mu0 = means[k * 3 + 0];
    const float mu1 = means[k * 3 + 1];
    const float mu2 = means[k * 3 + 2];
    
    // Compute difference
    const float d0 = x0 - mu0;
    const float d1 = x1 - mu1;
    const float d2 = x2 - mu2;
    
    // Load inverse Cholesky factor (lower triangular)
    // L_inv^{-1} * diff = y, so we compute y = L_inv * diff directly
    const float* L = &L_inv[k * 9];  // 3x3 matrix in row-major
    
    // Solve L * y = diff (L is lower triangular)
    const float y0 = d0 / L[0];                           // L[0,0]
    const float y1 = (d1 - L[3] * y0) / L[4];            // L[1,0], L[1,1]
    const float y2 = (d2 - L[6] * y0 - L[7] * y1) / L[8]; // L[2,0], L[2,1], L[2,2]
    
    // Mahalanobis distance: ||y||^2
    const float mahal = y0*y0 + y1*y1 + y2*y2;
    
    // Gaussian value
    const float amp = amplitudes[k];
    const float val = amp * expf(-0.5f * mahal);
    
    // Write output
    output[idx] = val;
}


// CUDA kernel for backward pass
__global__ void gaussian_eval_backward_kernel(
    const float* __restrict__ grad_output,  // (N, K) gradient from upstream
    const float* __restrict__ x,            // (N, 3) query points
    const float* __restrict__ means,        // (K, 3) centers
    const float* __restrict__ L_inv,        // (K, 3, 3) Cholesky factors
    const float* __restrict__ amplitudes,   // (K,) amplitudes
    const float* __restrict__ vals,         // (N, K) forward output
    float* __restrict__ grad_x,             // (N, 3) gradient w.r.t. x
    float* __restrict__ grad_means,         // (K, 3) gradient w.r.t. means
    float* __restrict__ grad_amplitudes,    // (K,) gradient w.r.t. amplitudes
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K;
    
    if (idx >= total) return;
    
    const int n = idx / K;
    const int k = idx % K;
    
    const float grad_out = grad_output[idx];
    const float val = vals[idx];
    const float amp = amplitudes[k];
    
    // Gradient w.r.t. amplitude
    if (amp > 1e-12f) {
        atomicAdd(&grad_amplitudes[k], grad_out * val / amp);
    }
    
    // Recompute forward quantities
    const float x0 = x[n * 3 + 0];
    const float x1 = x[n * 3 + 1];
    const float x2 = x[n * 3 + 2];
    
    const float mu0 = means[k * 3 + 0];
    const float mu1 = means[k * 3 + 1];
    const float mu2 = means[k * 3 + 2];
    
    const float d0 = x0 - mu0;
    const float d1 = x1 - mu1;
    const float d2 = x2 - mu2;
    
    const float* L = &L_inv[k * 9];
    
    const float y0 = d0 / L[0];
    const float y1 = (d1 - L[3] * y0) / L[4];
    const float y2 = (d2 - L[6] * y0 - L[7] * y1) / L[8];
    
    // Gradient w.r.t. Mahalanobis distance
    const float grad_mahal = grad_out * val * (-0.5f);
    
    // Gradient w.r.t. y
    const float grad_y0 = grad_mahal * 2.0f * y0;
    const float grad_y1 = grad_mahal * 2.0f * y1;
    const float grad_y2 = grad_mahal * 2.0f * y2;
    
    // Backprop through triangular solve: L^T @ grad_diff = grad_y
    // Solve L^T * grad_d = grad_y (upper triangular)
    const float grad_d2 = grad_y2 / L[8];
    const float grad_d1 = (grad_y1 - L[7] * grad_d2) / L[4];
    const float grad_d0 = (grad_y0 - L[3] * grad_d1 - L[6] * grad_d2) / L[0];
    
    // Gradient w.r.t. x (accumulate across K)
    atomicAdd(&grad_x[n * 3 + 0], grad_d0);
    atomicAdd(&grad_x[n * 3 + 1], grad_d1);
    atomicAdd(&grad_x[n * 3 + 2], grad_d2);
    
    // Gradient w.r.t. means
    atomicAdd(&grad_means[k * 3 + 0], -grad_d0);
    atomicAdd(&grad_means[k * 3 + 1], -grad_d1);
    atomicAdd(&grad_means[k * 3 + 2], -grad_d2);
}


// C++ interface
torch::Tensor gaussian_eval_forward_cuda(
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    const int N = x.size(0);
    const int K = means.size(0);
    
    auto output = torch::zeros({N, K}, x.options());
    
    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;
    
    gaussian_eval_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        output.data_ptr<float>(),
        N, K
    );
    
    return output;
}


std::vector<torch::Tensor> gaussian_eval_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes,
    torch::Tensor vals
) {
    const int N = x.size(0);
    const int K = means.size(0);
    
    auto grad_x = torch::zeros_like(x);
    auto grad_means = torch::zeros_like(means);
    auto grad_amplitudes = torch::zeros_like(amplitudes);
    
    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;
    
    gaussian_eval_backward_kernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        vals.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_amplitudes.data_ptr<float>(),
        N, K
    );
    
    return {grad_x, grad_means, torch::Tensor(), grad_amplitudes};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_eval_forward_cuda, "Gaussian evaluation forward (CUDA)");
    m.def("backward", &gaussian_eval_backward_cuda, "Gaussian evaluation backward (CUDA)");
}

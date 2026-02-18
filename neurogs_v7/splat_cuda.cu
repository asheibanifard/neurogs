#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

// ============================================================================
// CUDA Kernels for Alpha-compositing Splatting
// ============================================================================
//
// Fuses the inner loop of Gaussian splatting:
//   For each pixel (N pixels), loop over K sorted Gaussians:
//     1. Evaluate 2D Gaussian: exp(-0.5 * Mahalanobis)
//     2. Compute alpha = opacity * gauss_val
//     3. Front-to-back compositing: color += T * alpha * c_k; T *= (1 - alpha)
//
// This avoids materialising the (N, K) tensor which causes OOM.
// Backward uses the same 2-pass approach as 3D Gaussian Splatting.
// ============================================================================

#define BLOCK_SIZE 256


// ============================================================================
// Forward kernel: one thread per pixel
// ============================================================================
__global__ void splat_alpha_forward_kernel(
    const float* __restrict__ means_2d,    // (K, 2) sorted by depth
    const float* __restrict__ cov_inv,     // (K, 3) packed [inv_a, inv_b, inv_d]
    const float* __restrict__ opacities,   // (K,) sorted
    const float* __restrict__ colors,      // (K,) sorted (grayscale)
    const float* __restrict__ pixels,      // (N, 2)
    float* __restrict__ rendered,          // (N,) output
    float* __restrict__ T_out,             // (N,) final transmittance
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    float T = 1.0f;
    float C = 0.0f;

    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float ia = cov_inv[k * 3 + 0];
        const float ib = cov_inv[k * 3 + 1];
        const float id = cov_inv[k * 3 + 2];

        const float mahal = ia * dx * dx + 2.0f * ib * dx * dy + id * dy * dy;
        if (mahal > 16.0f) continue;  // exp(-8) ≈ 3e-4

        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;

        C += T * alpha * colors[k];
        T *= (1.0f - alpha);

        if (T < 1e-4f) break;
    }

    rendered[n] = C;
    T_out[n] = T;
}


// ============================================================================
// Backward kernel: one thread per pixel, 2-pass (fwd replay + grad accum)
// ============================================================================
// Gradients:
//   grad_means_2d[k]  += sum_n dL/d(means_2d[k]) from pixel n
//   grad_cov_inv[k]   += sum_n dL/d(cov_inv[k])   from pixel n
//   grad_opacities[k] += sum_n dL/d(opacity[k])    from pixel n
//   grad_colors[k]    += sum_n dL/d(color[k])      from pixel n
//
// Uses atomicAdd to accumulate across pixels.
//
__global__ void splat_alpha_backward_kernel(
    const float* __restrict__ dL_drendered,  // (N,) upstream gradient
    const float* __restrict__ means_2d,      // (K, 2)
    const float* __restrict__ cov_inv,       // (K, 3)
    const float* __restrict__ opacities,     // (K,)
    const float* __restrict__ colors,        // (K,)
    const float* __restrict__ pixels,        // (N, 2)
    float* __restrict__ grad_means_2d,       // (K, 2) output
    float* __restrict__ grad_cov_inv,        // (K, 3) output
    float* __restrict__ grad_opacities,      // (K,) output
    float* __restrict__ grad_colors,         // (K,) output
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float dL_dC = dL_drendered[n];
    if (fabsf(dL_dC) < 1e-12f) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    // Pass 1: forward replay to compute C_total
    float T = 1.0f;
    float C_total = 0.0f;
    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];
        const float ia = cov_inv[k * 3 + 0];
        const float ib = cov_inv[k * 3 + 1];
        const float id = cov_inv[k * 3 + 2];
        const float mahal = ia * dx * dx + 2.0f * ib * dx * dy + id * dy * dy;
        if (mahal > 16.0f) continue;
        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;
        C_total += T * alpha * colors[k];
        T *= (1.0f - alpha);
        if (T < 1e-4f) break;
    }

    // Pass 2: backward with gradient accumulation
    T = 1.0f;
    float S = 0.0f;  // accumulated color so far

    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];
        const float ia = cov_inv[k * 3 + 0];
        const float ib = cov_inv[k * 3 + 1];
        const float id = cov_inv[k * 3 + 2];
        const float mahal = ia * dx * dx + 2.0f * ib * dx * dy + id * dy * dy;
        if (mahal > 16.0f) continue;
        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;

        const float c_k = colors[k];
        const float w = T * alpha;

        // --- dL/d(color_k) = dL/dC * T * alpha ---
        atomicAdd(&grad_colors[k], dL_dC * w);

        // --- dL/d(alpha_k) using standard compositing backward ---
        // C = sum_j T_j * alpha_j * c_j
        // dC/d(alpha_k) = T_k * c_k + sum_{j>k} dT_j/d(alpha_k) * alpha_j * c_j
        // dT_j/d(alpha_k) = -T_j / (1 - alpha_k)  for j > k
        // So: dC/d(alpha_k) = T_k * c_k - (C_remaining) / (1 - alpha_k)
        // where C_remaining = C_total - S - w*c_k
        const float C_remaining = C_total - S - w * c_k;
        const float one_minus_alpha = fmaxf(1.0f - alpha, 0.001f);
        const float dC_dalpha = T * c_k - C_remaining / one_minus_alpha;
        const float dL_dalpha = dL_dC * dC_dalpha;

        // alpha = min(opacity * gauss, 0.999)
        // d(alpha)/d(opacity) = gauss  (if alpha < 0.999)
        // d(alpha)/d(gauss) = opacity  (if alpha < 0.999)
        const float alpha_unclamped = opacities[k] * gauss;
        const float clamp_grad = (alpha_unclamped < 0.999f) ? 1.0f : 0.0f;

        const float dL_dopacity = dL_dalpha * gauss * clamp_grad;
        const float dL_dgauss = dL_dalpha * opacities[k] * clamp_grad;

        atomicAdd(&grad_opacities[k], dL_dopacity);

        // gauss = exp(-0.5 * mahal), dL/d(mahal) = dL/d(gauss) * (-0.5) * gauss
        const float dL_dmahal = dL_dgauss * (-0.5f) * gauss;

        // mahal = ia*dx^2 + 2*ib*dx*dy + id*dy^2
        // d/d(ia) = dx^2, d/d(ib) = 2*dx*dy, d/d(id) = dy^2
        atomicAdd(&grad_cov_inv[k * 3 + 0], dL_dmahal * dx * dx);
        atomicAdd(&grad_cov_inv[k * 3 + 1], dL_dmahal * 2.0f * dx * dy);
        atomicAdd(&grad_cov_inv[k * 3 + 2], dL_dmahal * dy * dy);

        // d(mahal)/d(dx) = 2*ia*dx + 2*ib*dy, d(mahal)/d(dy) = 2*ib*dx + 2*id*dy
        // dx = px - mean_x, so d(mahal)/d(mean_x) = -(2*ia*dx + 2*ib*dy)
        const float dL_ddx = dL_dmahal * (2.0f * ia * dx + 2.0f * ib * dy);
        const float dL_ddy = dL_dmahal * (2.0f * ib * dx + 2.0f * id * dy);
        atomicAdd(&grad_means_2d[k * 2 + 0], -dL_ddx);
        atomicAdd(&grad_means_2d[k * 2 + 1], -dL_ddy);

        S += w * c_k;
        T *= one_minus_alpha;
        if (T < 1e-4f) break;
    }
}


// ============================================================================
// C++ interface functions
// ============================================================================

std::vector<torch::Tensor> splat_alpha_forward(
    torch::Tensor means_2d,     // (K, 2) sorted by depth
    torch::Tensor cov_inv,      // (K, 3) [inv_a, inv_b, inv_d]
    torch::Tensor opacities,    // (K,)
    torch::Tensor colors,       // (K,) grayscale
    torch::Tensor pixels        // (N, 2)
) {
    means_2d = means_2d.contiguous();
    cov_inv = cov_inv.contiguous();
    opacities = opacities.contiguous();
    colors = colors.contiguous();
    pixels = pixels.contiguous();

    const int K = means_2d.size(0);
    const int N = pixels.size(0);

    auto rendered = torch::zeros({N}, pixels.options());
    auto T_out = torch::ones({N}, pixels.options());

    if (K > 0 && N > 0) {
        const int threads = BLOCK_SIZE;
        const int blocks = (N + threads - 1) / threads;
        splat_alpha_forward_kernel<<<blocks, threads>>>(
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            opacities.data_ptr<float>(),
            colors.data_ptr<float>(),
            pixels.data_ptr<float>(),
            rendered.data_ptr<float>(),
            T_out.data_ptr<float>(),
            N, K
        );
    }

    return {rendered, T_out};
}


std::vector<torch::Tensor> splat_alpha_backward(
    torch::Tensor dL_drendered,  // (N,)
    torch::Tensor means_2d,      // (K, 2)
    torch::Tensor cov_inv,       // (K, 3)
    torch::Tensor opacities,     // (K,)
    torch::Tensor colors,        // (K,)
    torch::Tensor pixels         // (N, 2)
) {
    dL_drendered = dL_drendered.contiguous();
    means_2d = means_2d.contiguous();
    cov_inv = cov_inv.contiguous();
    opacities = opacities.contiguous();
    colors = colors.contiguous();
    pixels = pixels.contiguous();

    const int K = means_2d.size(0);
    const int N = pixels.size(0);

    auto grad_means_2d = torch::zeros({K, 2}, means_2d.options());
    auto grad_cov_inv = torch::zeros({K, 3}, cov_inv.options());
    auto grad_opacities = torch::zeros({K}, opacities.options());
    auto grad_colors = torch::zeros({K}, colors.options());

    if (K > 0 && N > 0) {
        const int threads = BLOCK_SIZE;
        const int blocks = (N + threads - 1) / threads;
        splat_alpha_backward_kernel<<<blocks, threads>>>(
            dL_drendered.data_ptr<float>(),
            means_2d.data_ptr<float>(),
            cov_inv.data_ptr<float>(),
            opacities.data_ptr<float>(),
            colors.data_ptr<float>(),
            pixels.data_ptr<float>(),
            grad_means_2d.data_ptr<float>(),
            grad_cov_inv.data_ptr<float>(),
            grad_opacities.data_ptr<float>(),
            grad_colors.data_ptr<float>(),
            N, K
        );
    }

    return {grad_means_2d, grad_cov_inv, grad_opacities, grad_colors};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &splat_alpha_forward, "Alpha-compositing splat forward (CUDA)");
    m.def("backward", &splat_alpha_backward, "Alpha-compositing splat backward (CUDA)");
}

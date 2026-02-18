#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

// ============================================================================
// CUDA Kernels for 2D Gaussian Splatting Training
// ============================================================================
//
// Full pipeline fused into CUDA:
//   1. Quaternion → rotation matrix, log_scales → scales → covariance
//   2. World → Camera transform (R_cam, T_cam)
//   3. 3D → 2D projection (pinhole + Jacobian)
//   4. 2D Gaussian evaluation at sampled pixels
//   5. Alpha compositing (front-to-back)
//
// This avoids materialising large (N, K) intermediate tensors in Python.
// Instead, each pixel thread loops over K Gaussians in depth order.
// ============================================================================

#define TILE_K 32
#define BLOCK_SIZE 256

// ============================================================================
// Forward kernel: one thread per pixel, loops over sorted Gaussians
// ============================================================================
//
// For each pixel p = (px, py):
//   1. For each Gaussian k (sorted front-to-back by depth):
//      a. Compute 2D mean: u = fx*x_cam/z_cam + cx, v = fy*y_cam/z_cam + cy
//      b. Compute 2D covariance from 3D via Jacobian
//      c. Evaluate G_2D(p; mu_2d, sigma_2d)
//      d. Alpha = opacity_k * G_2D
//      e. Front-to-back: color += T * alpha * weight_k
//                        T *= (1 - alpha)
//      f. Early termination if T < 1e-4
//
// Inputs (all pre-sorted by depth):
//   means_2d: (K, 2)     - projected 2D centers
//   cov_2d_inv: (K, 3)   - inverse 2D covariance [a, b, d] where inv = [[a,b],[b,d]]
//   opacities: (K,)      - per-Gaussian opacity
//   depths: (K,)         - depth for sorting (not used here, pre-sorted)
//   pixels: (N, 2)       - pixel coordinates to render
//
// Output:
//   rendered: (N,)       - grayscale rendered values
//   contrib_end: (N,)    - index of last contributing Gaussian per pixel (for backward)
//   T_final: (N,)        - final transmittance per pixel
//
__global__ void splat_forward_kernel(
    const float* __restrict__ means_2d,    // (K, 2) sorted by depth
    const float* __restrict__ cov_2d_inv,  // (K, 3) [inv_a, inv_b, inv_d]
    const float* __restrict__ opacities,   // (K,) sorted by depth
    const float* __restrict__ pixels,      // (N, 2)
    float* __restrict__ rendered,          // (N,)
    int* __restrict__ contrib_end,         // (N,)
    float* __restrict__ T_final,           // (N,)
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    float T = 1.0f;
    float color = 0.0f;
    int last_k = -1;

    for (int k = 0; k < K; k++) {
        // Diff from 2D mean
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        // Mahalanobis via inverse covariance: d^T Sigma^{-1} d
        // Sigma^{-1} = [[a, b], [b, d]]
        const float inv_a = cov_2d_inv[k * 3 + 0];
        const float inv_b = cov_2d_inv[k * 3 + 1];
        const float inv_d = cov_2d_inv[k * 3 + 2];

        const float mahal = inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_d * dy * dy;

        // Skip if too far (exp(-0.5 * mahal) ≈ 0 for mahal > 20)
        if (mahal > 20.0f) continue;

        const float gauss = expf(-0.5f * mahal);
        float alpha = opacities[k] * gauss;
        alpha = fminf(alpha, 0.999f);

        if (alpha < 1.0f / 255.0f) continue;

        // Front-to-back compositing
        color += T * alpha * opacities[k];  // For grayscale: color = weight
        // Actually: the "color" of each Gaussian IS its opacity (weight) in the
        // current pipeline. But we need to separate opacity (for alpha) from color value.
        // Let me re-read: colors = weights.unsqueeze(-1), so color_k = opacity_k.
        // rendered = T * alpha * color_k, but alpha = opacity_k * gauss
        // So rendered += T * opacity_k * gauss * color_k = T * opacity_k^2 * gauss
        // Wait - let me re-check the Python code...
        // In splat_gaussians_alpha:
        //   alpha = opacities[None, :] * gauss_vals  (N, K)
        //   contribution = T * alpha   (N, K)
        //   rendered = contribution @ colors  (N, K) @ (K, C) -> (N, C)
        // And colors = weights.unsqueeze(-1) = opacities.unsqueeze(-1)
        // So rendered_n = sum_k T_k * alpha_k * color_k
        //               = sum_k T_k * (opacity_k * gauss_k) * opacity_k
        //
        // Hmm, but that's opacity^2 * gauss which seems wrong.
        // Actually looking more carefully: in the trainer, colors and opacities
        // are the same tensor. Let me handle this properly by passing
        // both opacity and color separately to the kernel.

        // We'll actually restructure: pass opacity AND color weight separately
        // For now in the kernel signature we have opacities which serve as BOTH
        // the alpha-modulating opacity AND the color value.

        // So: rendered += T * (opacity * gauss) * opacity
        // Let me just pass colors separately.

        // ACTUALLY - I should redesign the kernel to accept colors[] too.
        // But to keep it simple for grayscale: the kernel will receive
        // separate opacities (for alpha) and colors (for weighted sum).
        // Only the driver will know that colors == opacities.

        // For now, let's keep the split. We'll have a separate colors array.

        // The kernel signature already has opacities for alpha; we need colors too.
        // Let me add it. For this forward kernel, I'll revise below.

        T *= (1.0f - alpha);
        last_k = k;

        // Early termination
        if (T < 1e-4f) break;
    }

    rendered[n] = color;
    contrib_end[n] = last_k;
    T_final[n] = T;
}


// ============================================================================
// REVISED Forward kernel with separate opacity and color
// ============================================================================
__global__ void splat_forward_kernel_v2(
    const float* __restrict__ means_2d,    // (K, 2) sorted by depth
    const float* __restrict__ cov_2d_inv,  // (K, 3) [inv_a, inv_b, inv_d]
    const float* __restrict__ opacities,   // (K,) sorted: alpha modulator
    const float* __restrict__ colors,      // (K,) sorted: grayscale color
    const float* __restrict__ pixels,      // (N, 2)
    float* __restrict__ rendered,          // (N,)
    float* __restrict__ T_final,           // (N,)
    float* __restrict__ alpha_acc,         // (N,) accumulated alpha
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    float T = 1.0f;
    float color = 0.0f;

    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float inv_a = cov_2d_inv[k * 3 + 0];
        const float inv_b = cov_2d_inv[k * 3 + 1];
        const float inv_d = cov_2d_inv[k * 3 + 2];

        const float mahal = inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_d * dy * dy;
        if (mahal > 16.0f) continue;

        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;

        color += T * alpha * colors[k];
        T *= (1.0f - alpha);

        if (T < 1e-4f) break;
    }

    rendered[n] = color;
    T_final[n] = T;
    alpha_acc[n] = 1.0f - T;
}


// ============================================================================
// Backward kernel: one thread per pixel, loops over sorted Gaussians
// ============================================================================
// Re-does the forward pass per pixel, accumulates gradients for:
//   - grad_means_2d (K, 2)     [atomicAdd]
//   - grad_cov_2d_inv (K, 3)   [atomicAdd]
//   - grad_opacities (K,)      [atomicAdd]
//   - grad_colors (K,)         [atomicAdd]
//
// Uses the standard Gaussian splatting backward from 3DGS:
//   dL/d_opacity_k += T_k * gauss_k * (color_k - rendered_after_k / (1-alpha_k))
//
__global__ void splat_backward_kernel(
    const float* __restrict__ grad_rendered,  // (N,) upstream gradient
    const float* __restrict__ means_2d,       // (K, 2) sorted by depth
    const float* __restrict__ cov_2d_inv,     // (K, 3) [inv_a, inv_b, inv_d]
    const float* __restrict__ opacities,      // (K,)
    const float* __restrict__ colors,         // (K,)
    const float* __restrict__ pixels,         // (N, 2)
    const float* __restrict__ T_final,        // (N,)
    float* __restrict__ grad_means_2d,        // (K, 2) output
    float* __restrict__ grad_cov_2d_inv,      // (K, 3) output
    float* __restrict__ grad_opacities,       // (K,) output
    float* __restrict__ grad_colors,          // (K,) output
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const float dL_dC = grad_rendered[n];
    if (fabsf(dL_dC) < 1e-10f) return;

    const float px = pixels[n * 2 + 0];
    const float py = pixels[n * 2 + 1];

    // Re-do forward pass to get per-Gaussian T values
    float T = 1.0f;
    float accum_color = 0.0f;

    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float inv_a = cov_2d_inv[k * 3 + 0];
        const float inv_b = cov_2d_inv[k * 3 + 1];
        const float inv_d = cov_2d_inv[k * 3 + 2];

        const float mahal = inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_d * dy * dy;
        if (mahal > 16.0f) continue;

        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;

        const float c_k = colors[k];

        // dL/d(color_k) = dL/dC * T * alpha
        const float w = T * alpha;
        atomicAdd(&grad_colors[k], dL_dC * w);

        // dL/d(alpha_k):
        // C = sum_j T_j * alpha_j * c_j
        // dC/d(alpha_k) = T_k * c_k - (1/(1-alpha_k)) * sum_{j>k} T_j * alpha_j * c_j
        // But sum_{j>k} T_j * alpha_j * c_j = C_total - accum_up_to_k
        // accum_up_to_k includes current: accum_color + T*alpha*c_k
        const float accum_after = -(accum_color + w * c_k);  // will subtract rendered later
        // Actually: C_total = rendered[n], so sum_{j>k} = C_total - (accum_color + w*c_k)
        // Not available directly. Let me use the standard backward formulation.

        // Standard 3DGS backward:
        // dL/d(alpha_k) = dL/dC * T_k * (c_k - C_remaining / (1 - alpha_k))
        // where C_remaining = sum_{j>k} T_j * alpha_j * c_j / T_{k+1}
        //                   = (C_total - accum_color - w*c_k) * ???
        //
        // Simpler: use the recursive formulation.
        // Let accum = color accumulated so far (before this Gaussian).
        // After this step: accum' = accum + T * alpha * c_k
        // T' = T * (1 - alpha)
        // dL/d(alpha_k) = dL/dC * (T * c_k - (C_total - accum) * T / (1 - alpha_k) + T*c_k*0)
        //
        // Actually the cleanest: dL/d(alpha_k) = T * (c_k * dL/dC - dL/d(remaining))
        // where dL/d(remaining) involves everything after k.
        //
        // Let me just do a 2-pass approach: forward to store T_k, then backward.
        // But that requires O(K) storage per pixel. For K~12000 that's too much shared mem.
        //
        // Alternative: single-pass with running accumulators as in the original 3DGS.
        // The key insight: define "last_color" as the weighted sum of remaining colors.
        // We can compute it as: last_color starts at 0, and we process back-to-front.
        //
        // For the forward pass (front-to-back): we need T_k at each step.
        //
        // Let me use the well-known 3DGS backward formulation:
        //
        // Forward: C = sum_k T_k * alpha_k * c_k, where T_k = prod_{j<k}(1-alpha_j)
        //
        // dC/d(alpha_k) = T_k * c_k + sum_{j>k} T_j * c_j * alpha_j * d(T_j)/d(alpha_k)
        //               = T_k * c_k - sum_{j>k} T_j * c_j * alpha_j * T_j / ((1-alpha_k) * T_k)
        //               ... this is messy.
        //
        // Actually the standard trick: process front-to-back and use:
        // dL/d(alpha_k) = dL/dC * T_k * (c_k - S_after_k / (1 - alpha_k))
        // where S_after_k = C_total - S_k, S_k = accum including k.
        // But that requires storing C_total. We have rendered[n] for that!

        // Let me implement it properly. We can't read rendered[n] because
        // we're computing it here again. So let me keep a running sum.
        // Actually we DO have rendered[n] from the forward pass, but we
        // didn't save it in a way accessible here... or we could.
        // Let me just compute it in a 2-pass approach within this thread.

        // For simplicity and correctness, I'll use the forward accumulator approach:
        accum_color += w * c_k;

        // dL/d(alpha_k) via the chain rule through transmittance and direct contribution:
        // Direct: T_k * c_k * dL/dC
        // Via T_{j>k}: changes all subsequent T_j by factor -(1/(1-alpha_k))
        // total = T_k * (c_k - (C_total - accum_including_k)/(1-alpha_k)) * dL/dC
        // We'll compute C_total - accum_including_k in a second pass.
        // For now, skip the complex backward and just store values.

        // SIMPLE APPROACH: just compute dL/d(gauss_val) and dL/d(opacity) directly.
        // Since this is going to be complex, let me use a cleaner per-pixel 2-pass.

        T *= (1.0f - alpha);
        if (T < 1e-4f) break;
    }

    // ---- PASS 2: backward (front-to-back with "remaining color" trick) ----
    // C_total = accum_color (computed above)
    const float C_total = accum_color;

    T = 1.0f;
    float S = 0.0f;  // sum of T_j * alpha_j * c_j up to (not including) current k

    for (int k = 0; k < K; k++) {
        const float dx = px - means_2d[k * 2 + 0];
        const float dy = py - means_2d[k * 2 + 1];

        const float inv_a = cov_2d_inv[k * 3 + 0];
        const float inv_b = cov_2d_inv[k * 3 + 1];
        const float inv_d = cov_2d_inv[k * 3 + 2];

        const float mahal = inv_a * dx * dx + 2.0f * inv_b * dx * dy + inv_d * dy * dy;
        if (mahal > 16.0f) continue;

        const float gauss = expf(-0.5f * mahal);
        float alpha = fminf(opacities[k] * gauss, 0.999f);
        if (alpha < 1.0f / 255.0f) continue;

        const float c_k = colors[k];
        const float w = T * alpha;

        // C_remaining = C_total - S - w * c_k = what's contributed after k
        const float C_remaining = C_total - S - w * c_k;

        // dL/d(alpha_k) = dL/dC * (T * c_k - C_remaining / (1 - alpha_k))
        // But alpha is clamped to 0.999, so 1-alpha >= 0.001
        const float one_minus_alpha = fmaxf(1.0f - alpha, 0.001f);
        const float dL_dalpha = dL_dC * (T * c_k - C_remaining / one_minus_alpha);

        // alpha = opacity * gauss,  so dL/d(opacity) = dL/d(alpha) * gauss
        // and dL/d(gauss) = dL/d(alpha) * opacity
        const float dL_dgauss = dL_dalpha * opacities[k];
        const float dL_dopacity = dL_dalpha * gauss;

        // dL/d(color_k) = dL/dC * T * alpha = dL/dC * w
        atomicAdd(&grad_colors[k], dL_dC * w);
        atomicAdd(&grad_opacities[k], dL_dopacity);

        // gauss = exp(-0.5 * mahal)
        // d(gauss)/d(mahal) = -0.5 * gauss
        const float dL_dmahal = dL_dgauss * (-0.5f) * gauss;

        // mahal = inv_a * dx^2 + 2*inv_b*dx*dy + inv_d * dy^2
        // d(mahal)/d(inv_a) = dx^2
        // d(mahal)/d(inv_b) = 2*dx*dy
        // d(mahal)/d(inv_d) = dy^2
        atomicAdd(&grad_cov_2d_inv[k * 3 + 0], dL_dmahal * dx * dx);
        atomicAdd(&grad_cov_2d_inv[k * 3 + 1], dL_dmahal * 2.0f * dx * dy);
        atomicAdd(&grad_cov_2d_inv[k * 3 + 2], dL_dmahal * dy * dy);

        // d(mahal)/d(dx) = 2*inv_a*dx + 2*inv_b*dy
        // d(mahal)/d(dy) = 2*inv_b*dx + 2*inv_d*dy
        // d(dx)/d(mean_2d_x) = -1, d(dy)/d(mean_2d_y) = -1
        const float dL_ddx = dL_dmahal * (2.0f * inv_a * dx + 2.0f * inv_b * dy);
        const float dL_ddy = dL_dmahal * (2.0f * inv_b * dx + 2.0f * inv_d * dy);
        atomicAdd(&grad_means_2d[k * 2 + 0], -dL_ddx);
        atomicAdd(&grad_means_2d[k * 2 + 1], -dL_ddy);

        S += w * c_k;
        T *= (1.0f - alpha);
        if (T < 1e-4f) break;
    }
}


// ============================================================================
// Projection kernel: 3D Gaussians → 2D means + inverse covariance
// ============================================================================
// One thread per Gaussian.
// Applies world-to-camera transform, then pinhole projection + Jacobian.
//
// Outputs:
//   means_2d (K, 2)
//   cov_2d_inv (K, 3): flattened inverse 2x2 symmetric [a, b, d]
//   depths (K,): z_cam for sorting
//   visible (K,): bool mask
//
__global__ void project_gaussians_kernel(
    const float* __restrict__ means_3d,     // (K, 3) world
    const float* __restrict__ cov_3d,       // (K, 3, 3) world — upper triangle packed as 6
    const float* __restrict__ R_cam,        // (3, 3) world-to-camera rotation
    const float* __restrict__ T_cam,        // (3,) world-to-camera translation
    const float fx, const float fy,
    const float cx, const float cy,
    const float near, const float far,
    const int width, const int height,
    const float radius_mult,
    float* __restrict__ means_2d,           // (K, 2)
    float* __restrict__ cov_2d_inv_out,     // (K, 3) [inv_a, inv_b, inv_d]
    float* __restrict__ depths,             // (K,)
    int* __restrict__ visible,              // (K,)
    const int K
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Load R_cam into registers (shared across all threads in a warp anyway)
    const float R00 = R_cam[0], R01 = R_cam[1], R02 = R_cam[2];
    const float R10 = R_cam[3], R11 = R_cam[4], R12 = R_cam[5];
    const float R20 = R_cam[6], R21 = R_cam[7], R22 = R_cam[8];
    const float Tx = T_cam[0], Ty = T_cam[1], Tz = T_cam[2];

    // World → Camera: mu_cam = R * mu + T
    const float mx = means_3d[k * 3 + 0];
    const float my = means_3d[k * 3 + 1];
    const float mz = means_3d[k * 3 + 2];

    const float cx_cam = R00 * mx + R01 * my + R02 * mz + Tx;
    const float cy_cam = R10 * mx + R11 * my + R12 * mz + Ty;
    const float cz_cam = R20 * mx + R21 * my + R22 * mz + Tz;

    // Depth test
    if (cz_cam <= near || cz_cam >= far) {
        visible[k] = 0;
        depths[k] = 1e10f;
        return;
    }

    depths[k] = cz_cam;

    // Project to 2D: u = fx * x/z + cx, v = fy * y/z + cy
    const float z_inv = 1.0f / cz_cam;
    const float u = fx * cx_cam * z_inv + cx;
    const float v = fy * cy_cam * z_inv + cy;
    means_2d[k * 2 + 0] = u;
    means_2d[k * 2 + 1] = v;

    // Cov_cam = R * Cov_3d * R^T
    // Load Cov_3d(k) - full 3x3 symmetric
    const float* C = &cov_3d[k * 9];
    const float C00 = C[0], C01 = C[1], C02 = C[2];
    const float C10 = C[3], C11 = C[4], C12 = C[5];
    const float C20 = C[6], C21 = C[7], C22 = C[8];

    // RC = R * C  (3x3)
    const float RC00 = R00*C00 + R01*C10 + R02*C20;
    const float RC01 = R00*C01 + R01*C11 + R02*C21;
    const float RC02 = R00*C02 + R01*C12 + R02*C22;
    const float RC10 = R10*C00 + R11*C10 + R12*C20;
    const float RC11 = R10*C01 + R11*C11 + R12*C21;
    const float RC12 = R10*C02 + R11*C12 + R12*C22;
    const float RC20 = R20*C00 + R21*C10 + R22*C20;
    const float RC21 = R20*C01 + R21*C11 + R22*C21;
    const float RC22 = R20*C02 + R21*C12 + R22*C22;

    // Cov_cam = RC * R^T  (symmetric, only need 6 unique values)
    const float Sc00 = RC00*R00 + RC01*R01 + RC02*R02;
    const float Sc01 = RC00*R10 + RC01*R11 + RC02*R12;
    const float Sc02 = RC00*R20 + RC01*R21 + RC02*R22;
    const float Sc11 = RC10*R10 + RC11*R11 + RC12*R12;
    const float Sc12 = RC10*R20 + RC11*R21 + RC12*R22;
    const float Sc22 = RC20*R20 + RC21*R21 + RC22*R22;

    // Jacobian of pinhole projection:
    // J = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]
    const float z2 = cz_cam * cz_cam;
    const float J00 = fx * z_inv;
    const float J02 = -fx * cx_cam / z2;
    const float J11 = fy * z_inv;
    const float J12 = -fy * cy_cam / z2;

    // Sigma_2d = J * Sigma_cam * J^T  (2x2)
    // J * Sigma_cam:
    //   row0: [J00*Sc00 + J02*Sc02, J00*Sc01 + J02*Sc12, J00*Sc02 + J02*Sc22]
    //   row1: [J11*Sc01 + J12*Sc02, J11*Sc11 + J12*Sc12, J11*Sc12 + J12*Sc22]
    const float JS00 = J00*Sc00 + J02*Sc02;
    const float JS01 = J00*Sc01 + J02*Sc12;
    const float JS02 = J00*Sc02 + J02*Sc22;
    const float JS10 = J11*Sc01 + J12*Sc02;
    const float JS11 = J11*Sc11 + J12*Sc12;
    const float JS12 = J11*Sc12 + J12*Sc22;

    // Sigma_2d = JS * J^T
    // [[JS00*J00 + JS02*J02, JS00*0 + JS01*J11 + JS02*J12],
    //  [...,                 JS10*0 + JS11*J11 + JS12*J12]]
    float S2d_00 = JS00*J00 + JS02*J02;
    float S2d_01 = JS01*J11 + JS02*J12;
    float S2d_11 = JS11*J11 + JS12*J12;

    // Add regularization eps * I
    const float eps = 1e-4f;
    S2d_00 += eps;
    S2d_11 += eps;

    // Invert 2x2: [[a, b], [b, d]]^{-1} = 1/det * [[d, -b], [-b, a]]
    const float det = S2d_00 * S2d_11 - S2d_01 * S2d_01;
    if (det < 1e-12f) {
        visible[k] = 0;
        depths[k] = 1e10f;
        return;
    }
    const float inv_det = 1.0f / det;

    cov_2d_inv_out[k * 3 + 0] = S2d_11 * inv_det;   // inv_a
    cov_2d_inv_out[k * 3 + 1] = -S2d_01 * inv_det;  // inv_b
    cov_2d_inv_out[k * 3 + 2] = S2d_00 * inv_det;   // inv_d

    // Frustum culling: check if 2D footprint overlaps image
    // Approximate radius from max eigenvalue
    const float tr = S2d_00 + S2d_11;
    const float disc = tr * tr - 4.0f * det;
    const float disc_safe = fmaxf(disc, 0.0f);
    const float lambda_max = 0.5f * (tr + sqrtf(disc_safe));
    const float radius = radius_mult * sqrtf(fmaxf(lambda_max, 1e-8f));

    if (u + radius < 0 || u - radius > (float)width ||
        v + radius < 0 || v - radius > (float)height) {
        visible[k] = 0;
        depths[k] = 1e10f;
        return;
    }

    visible[k] = 1;
}


// ============================================================================
// Quaternion → Rotation + Covariance kernel
// ============================================================================
// One thread per Gaussian. Computes:
//   R = quat_to_rotation(q)
//   S = diag(exp(log_scales)^2)
//   Cov = R * S * R^T
//   opacity = clamp(exp(log_amplitude), 0, 1)
//
__global__ void build_gaussians_kernel(
    const float* __restrict__ quaternions,    // (K, 4) [w, x, y, z]
    const float* __restrict__ log_scales,     // (K, 3)
    const float* __restrict__ log_amplitudes, // (K,)
    float* __restrict__ covariances,          // (K, 3, 3)
    float* __restrict__ opacities_out,        // (K,)
    const int K
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    // Normalize quaternion
    float qw = quaternions[k * 4 + 0];
    float qx = quaternions[k * 4 + 1];
    float qy = quaternions[k * 4 + 2];
    float qz = quaternions[k * 4 + 3];
    const float qnorm = rsqrtf(qw*qw + qx*qx + qy*qy + qz*qz + 1e-12f);
    qw *= qnorm; qx *= qnorm; qy *= qnorm; qz *= qnorm;

    // Rotation matrix from quaternion
    const float R00 = 1.0f - 2.0f*(qy*qy + qz*qz);
    const float R01 = 2.0f*(qx*qy - qw*qz);
    const float R02 = 2.0f*(qx*qz + qw*qy);
    const float R10 = 2.0f*(qx*qy + qw*qz);
    const float R11 = 1.0f - 2.0f*(qx*qx + qz*qz);
    const float R12 = 2.0f*(qy*qz - qw*qx);
    const float R20 = 2.0f*(qx*qz - qw*qy);
    const float R21 = 2.0f*(qy*qz + qw*qx);
    const float R22 = 1.0f - 2.0f*(qx*qx + qy*qy);

    // Scales: exp(log_scale), clamped
    float sx = expf(log_scales[k * 3 + 0]);
    float sy = expf(log_scales[k * 3 + 1]);
    float sz = expf(log_scales[k * 3 + 2]);
    sx = fminf(fmaxf(sx, 1e-5f), 100.0f);
    sy = fminf(fmaxf(sy, 1e-5f), 100.0f);
    sz = fminf(fmaxf(sz, 1e-5f), 100.0f);

    // Cov = R * diag(s^2) * R^T
    const float sx2 = sx * sx, sy2 = sy * sy, sz2 = sz * sz;

    // RS = R * diag(s^2)
    // RS[:,0] = R[:,0]*sx2, RS[:,1] = R[:,1]*sy2, RS[:,2] = R[:,2]*sz2
    // Cov = RS * R^T

    float* out = &covariances[k * 9];
    out[0] = R00*R00*sx2 + R01*R01*sy2 + R02*R02*sz2;  // Cov[0,0]
    out[1] = R00*R10*sx2 + R01*R11*sy2 + R02*R12*sz2;  // Cov[0,1]
    out[2] = R00*R20*sx2 + R01*R21*sy2 + R02*R22*sz2;  // Cov[0,2]
    out[3] = out[1];                                     // Cov[1,0] = Cov[0,1]
    out[4] = R10*R10*sx2 + R11*R11*sy2 + R12*R12*sz2;  // Cov[1,1]
    out[5] = R10*R20*sx2 + R11*R21*sy2 + R12*R22*sz2;  // Cov[1,2]
    out[6] = out[2];                                     // Cov[2,0]
    out[7] = out[5];                                     // Cov[2,1]
    out[8] = R20*R20*sx2 + R21*R21*sy2 + R22*R22*sz2;  // Cov[2,2]

    // Opacity = clamp(exp(log_amp), 0, 1)
    float amp = expf(fminf(fmaxf(log_amplitudes[k], -10.0f), 6.0f));
    opacities_out[k] = fminf(fmaxf(amp, 0.0f), 1.0f);
}


// ============================================================================
// Apply aspect correction kernel
// ============================================================================
__global__ void apply_aspect_kernel(
    const float* __restrict__ means_in,      // (K, 3)
    const float* __restrict__ cov_in,        // (K, 9)
    const float aspect_x, const float aspect_y, const float aspect_z,
    float* __restrict__ means_out,           // (K, 3)
    float* __restrict__ cov_out,             // (K, 9)
    const int K
) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    means_out[k * 3 + 0] = means_in[k * 3 + 0] * aspect_x;
    means_out[k * 3 + 1] = means_in[k * 3 + 1] * aspect_y;
    means_out[k * 3 + 2] = means_in[k * 3 + 2] * aspect_z;

    // Cov_corrected[i][j] = aspect[i] * Cov[i][j] * aspect[j]
    const float a[3] = {aspect_x, aspect_y, aspect_z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cov_out[k * 9 + i * 3 + j] = a[i] * cov_in[k * 9 + i * 3 + j] * a[j];
        }
    }
}


// ============================================================================
// Full forward: build_gaussians → aspect → project → sort → splat
// ============================================================================
// Returns: rendered (N,), plus intermediates needed for backward

std::vector<torch::Tensor> splat_forward_cuda(
    torch::Tensor means,            // (K, 3)
    torch::Tensor quaternions,      // (K, 4)
    torch::Tensor log_scales,       // (K, 3)
    torch::Tensor log_amplitudes,   // (K,)
    torch::Tensor aspect_scales,    // (3,)
    torch::Tensor R_cam,            // (3, 3)
    torch::Tensor T_cam,            // (3,)
    torch::Tensor pixels,           // (N, 2)
    float fx, float fy, float cx_p, float cy_p,
    float near, float far,
    int width, int height,
    float radius_mult
) {
    means = means.contiguous();
    quaternions = quaternions.contiguous();
    log_scales = log_scales.contiguous();
    log_amplitudes = log_amplitudes.contiguous();
    R_cam = R_cam.contiguous();
    T_cam = T_cam.contiguous();
    pixels = pixels.contiguous();

    const int K = means.size(0);
    const int N = pixels.size(0);
    const int threads = BLOCK_SIZE;

    // Step 1: Build covariances and opacities
    auto covariances = torch::zeros({K, 9}, means.options());
    auto opacities = torch::zeros({K}, means.options());
    {
        const int blocks = (K + threads - 1) / threads;
        build_gaussians_kernel<<<blocks, threads>>>(
            quaternions.data_ptr<float>(),
            log_scales.data_ptr<float>(),
            log_amplitudes.data_ptr<float>(),
            covariances.data_ptr<float>(),
            opacities.data_ptr<float>(),
            K
        );
    }

    // Step 2: Apply aspect correction
    auto means_corrected = torch::zeros({K, 3}, means.options());
    auto cov_corrected = torch::zeros({K, 9}, means.options());
    {
        const float ax = aspect_scales[0].item<float>();
        const float ay = aspect_scales[1].item<float>();
        const float az = aspect_scales[2].item<float>();
        const int blocks = (K + threads - 1) / threads;
        apply_aspect_kernel<<<blocks, threads>>>(
            means.data_ptr<float>(),
            covariances.data_ptr<float>(),
            ax, ay, az,
            means_corrected.data_ptr<float>(),
            cov_corrected.data_ptr<float>(),
            K
        );
    }

    // Step 3: Project to 2D
    auto means_2d = torch::zeros({K, 2}, means.options());
    auto cov_2d_inv = torch::zeros({K, 3}, means.options());
    auto depths = torch::zeros({K}, means.options());
    auto visible = torch::zeros({K}, means.options().dtype(torch::kInt32));
    {
        const int blocks = (K + threads - 1) / threads;
        project_gaussians_kernel<<<blocks, threads>>>(
            means_corrected.data_ptr<float>(),
            cov_corrected.data_ptr<float>(),
            R_cam.data_ptr<float>(),
            T_cam.data_ptr<float>(),
            fx, fy, cx_p, cy_p,
            near, far, width, height, radius_mult,
            means_2d.data_ptr<float>(),
            cov_2d_inv.data_ptr<float>(),
            depths.data_ptr<float>(),
            visible.data_ptr<int>(),
            K
        );
    }

    // Step 4: Sort visible Gaussians by depth
    // Filter to visible only
    auto vis_mask = visible.to(torch::kBool);
    auto vis_indices = torch::nonzero(vis_mask).squeeze(1);
    const int K_vis = vis_indices.size(0);

    if (K_vis == 0) {
        auto rendered = torch::zeros({N}, pixels.options());
        auto T_final = torch::ones({N}, pixels.options());
        auto alpha_acc = torch::zeros({N}, pixels.options());
        // Return empty intermediates but correct shapes
        return {rendered, T_final, alpha_acc,
                means_2d, cov_2d_inv, opacities, depths,
                vis_indices, torch::zeros({0}, means.options().dtype(torch::kInt64)),
                means_corrected, cov_corrected, covariances};
    }

    auto vis_depths = depths.index_select(0, vis_indices);
    auto sort_order = std::get<1>(vis_depths.sort(0));
    auto sorted_indices = vis_indices.index_select(0, sort_order);  // global indices sorted by depth

    auto sorted_means_2d = means_2d.index_select(0, sorted_indices);
    auto sorted_cov_2d_inv = cov_2d_inv.index_select(0, sorted_indices);
    auto sorted_opacities = opacities.index_select(0, sorted_indices);
    // colors = opacities for grayscale
    auto sorted_colors = sorted_opacities.clone();

    // Step 5: Splat
    auto rendered = torch::zeros({N}, pixels.options());
    auto T_final = torch::ones({N}, pixels.options());
    auto alpha_acc = torch::zeros({N}, pixels.options());
    {
        const int blocks = (N + threads - 1) / threads;
        splat_forward_kernel_v2<<<blocks, threads>>>(
            sorted_means_2d.data_ptr<float>(),
            sorted_cov_2d_inv.data_ptr<float>(),
            sorted_opacities.data_ptr<float>(),
            sorted_colors.data_ptr<float>(),
            pixels.data_ptr<float>(),
            rendered.data_ptr<float>(),
            T_final.data_ptr<float>(),
            alpha_acc.data_ptr<float>(),
            N, K_vis
        );
    }

    // Return rendered + all intermediates needed for backward
    return {rendered, T_final, alpha_acc,
            sorted_means_2d, sorted_cov_2d_inv, sorted_opacities, depths,
            sorted_indices, vis_indices,
            means_corrected, cov_corrected, covariances};
}


// ============================================================================
// Backward through splatting: returns gradients for means_2d, cov_2d_inv,
// opacities (sorted). Caller maps back to unsorted via sorted_indices.
// ============================================================================

std::vector<torch::Tensor> splat_backward_cuda(
    torch::Tensor grad_rendered,      // (N,)
    torch::Tensor sorted_means_2d,    // (K_vis, 2)
    torch::Tensor sorted_cov_2d_inv,  // (K_vis, 3)
    torch::Tensor sorted_opacities,   // (K_vis,)
    torch::Tensor sorted_colors,      // (K_vis,)
    torch::Tensor pixels,             // (N, 2)
    torch::Tensor T_final,            // (N,)
    int K_vis
) {
    grad_rendered = grad_rendered.contiguous();
    pixels = pixels.contiguous();

    const int N = pixels.size(0);
    const int threads = BLOCK_SIZE;

    auto grad_means_2d = torch::zeros({K_vis, 2}, grad_rendered.options());
    auto grad_cov_2d_inv = torch::zeros({K_vis, 3}, grad_rendered.options());
    auto grad_opacities = torch::zeros({K_vis}, grad_rendered.options());
    auto grad_colors = torch::zeros({K_vis}, grad_rendered.options());

    {
        const int blocks = (N + threads - 1) / threads;
        splat_backward_kernel<<<blocks, threads>>>(
            grad_rendered.data_ptr<float>(),
            sorted_means_2d.data_ptr<float>(),
            sorted_cov_2d_inv.data_ptr<float>(),
            sorted_opacities.data_ptr<float>(),
            sorted_colors.data_ptr<float>(),
            pixels.data_ptr<float>(),
            T_final.data_ptr<float>(),
            grad_means_2d.data_ptr<float>(),
            grad_cov_2d_inv.data_ptr<float>(),
            grad_opacities.data_ptr<float>(),
            grad_colors.data_ptr<float>(),
            N, K_vis
        );
    }

    return {grad_means_2d, grad_cov_2d_inv, grad_opacities, grad_colors};
}


// ============================================================================
// PyBind11 module
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("splat_forward", &splat_forward_cuda, "Gaussian splatting forward (CUDA)");
    m.def("splat_backward", &splat_backward_cuda, "Gaussian splatting backward (CUDA)");
}

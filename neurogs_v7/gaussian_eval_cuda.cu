#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================
// Forward kernel: evaluate all Gaussians at all points
// ============================================================
// Each thread handles one (n, k) pair.
// Fused kernel avoids materialising large intermediate tensors.
__global__ void gaussian_eval_forward_kernel(
    const float* __restrict__ x,           // (N, 3)
    const float* __restrict__ means,       // (K, 3)
    const float* __restrict__ L_chol,      // (K, 3, 3) Cholesky factors (lower triangular)
    const float* __restrict__ amplitudes,  // (K,)
    float* __restrict__ output,            // (N, K)
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    // diff = x_n - mu_k
    const float d0 = x[n * 3 + 0] - means[k * 3 + 0];
    const float d1 = x[n * 3 + 1] - means[k * 3 + 1];
    const float d2 = x[n * 3 + 2] - means[k * 3 + 2];

    // L = L_chol[k] (3x3 lower triangular, row-major)
    const float* L = &L_chol[k * 9];

    // Forward substitution: solve L y = d
    const float y0 = d0 / L[0];
    const float y1 = (d1 - L[3] * y0) / L[4];
    const float y2 = (d2 - L[6] * y0 - L[7] * y1) / L[8];

    // Mahalanobis distance
    const float mahal = y0 * y0 + y1 * y1 + y2 * y2;

    output[idx] = amplitudes[k] * expf(-0.5f * mahal);
}


// ============================================================
// Backward kernel: grad_x, grad_means, grad_amplitudes, grad_L_chol
// ============================================================
// Now also computes grad_L_chol, so the Python side doesn't need to
// redo the expensive (N*K) solve_triangular.
//
// Per (n,k):
//   grad_mahal = grad_out * val * (-0.5)
//   grad_y_i   = grad_mahal * 2 * y_i
//
// Gradient through forward substitution L y = d:
//   grad_d via L^T solve (for grad_x, grad_means)
//   grad_L_{ij} = -grad_y_i * y_j / L_{ii}  (lower-triangular entries only)
//
// grad_L_chol is accumulated per-Gaussian across all N points via atomicAdd.
__global__ void gaussian_eval_backward_kernel(
    const float* __restrict__ grad_output,  // (N, K)
    const float* __restrict__ x,            // (N, 3)
    const float* __restrict__ means,        // (K, 3)
    const float* __restrict__ L_chol,       // (K, 3, 3)
    const float* __restrict__ amplitudes,   // (K,)
    const float* __restrict__ vals,         // (N, K)
    float* __restrict__ grad_x,             // (N, 3)
    float* __restrict__ grad_means,         // (K, 3)
    float* __restrict__ grad_amplitudes,    // (K,)
    float* __restrict__ grad_L,             // (K, 9)  — full 3x3 row-major, only lower-tri meaningful
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    const float go = grad_output[idx];
    const float val = vals[idx];
    const float amp = amplitudes[k];

    // --- grad_amplitudes ---
    if (amp > 1e-12f) {
        atomicAdd(&grad_amplitudes[k], go * val / amp);
    }

    // --- Recompute forward substitution ---
    const float d0 = x[n * 3 + 0] - means[k * 3 + 0];
    const float d1 = x[n * 3 + 1] - means[k * 3 + 1];
    const float d2 = x[n * 3 + 2] - means[k * 3 + 2];

    const float* L = &L_chol[k * 9];
    const float L00 = L[0], L10 = L[3], L11 = L[4];
    const float L20 = L[6], L21 = L[7], L22 = L[8];

    const float y0 = d0 / L00;
    const float y1 = (d1 - L10 * y0) / L11;
    const float y2 = (d2 - L20 * y0 - L21 * y1) / L22;

    // --- grad through Mahalanobis ---
    const float gm = go * val * (-0.5f);  // grad_mahal
    const float gy0 = gm * 2.0f * y0;
    const float gy1 = gm * 2.0f * y1;
    const float gy2 = gm * 2.0f * y2;

    // --- grad_d via backward substitution of L^T gd = gy ---
    const float gd2 = gy2 / L22;
    const float gd1 = (gy1 - L21 * gd2) / L11;
    const float gd0 = (gy0 - L10 * gd1 - L20 * gd2) / L00;

    // grad_x (accumulate across K)
    atomicAdd(&grad_x[n * 3 + 0], gd0);
    atomicAdd(&grad_x[n * 3 + 1], gd1);
    atomicAdd(&grad_x[n * 3 + 2], gd2);

    // grad_means = -grad_d
    atomicAdd(&grad_means[k * 3 + 0], -gd0);
    atomicAdd(&grad_means[k * 3 + 1], -gd1);
    atomicAdd(&grad_means[k * 3 + 2], -gd2);

    // --- grad_L_chol (lower triangular entries) ---
    // From L y = d, differentiating w.r.t. L_{ij} (j <= i):
    //   ∂(Ly)_i / ∂L_{ij} = y_j
    //   grad_L_{ij} = -grad_y_i * y_j / L_{ii}
    //
    // But we need to be more careful with the chain rule through
    // forward substitution.  For lower-triangular L and y = L^{-1}d:
    //
    //   y0 = d0 / L00
    //   y1 = (d1 - L10*y0) / L11
    //   y2 = (d2 - L20*y0 - L21*y1) / L22
    //
    // Direct differentiation:
    //   ∂y0/∂L00 = -y0/L00
    //   ∂y1/∂L10 = -y0/L11
    //   ∂y1/∂L11 = -y1/L11
    //   ∂y2/∂L20 = -y0/L22
    //   ∂y2/∂L21 = -y1/L22
    //   ∂y2/∂L22 = -y2/L22
    //
    // But y1 depends on y0, and y2 depends on y0, y1.
    // The total derivative accounts for indirect effects.
    //
    // grad_L_{ij} = Σ_m gy_m * ∂y_m/∂L_{ij}  (total derivative)
    //
    // For L00:  gy0 * (-y0/L00) + gy1 * (L10*y0/(L00*L11)) + gy2 * ...
    //
    // Simpler formulation: grad_L_{ij} = -(L^{-T} (gy ⊗ y))_{ij}
    // Which for 3x3 lower triangular is:
    //
    // Using the gd (= L^{-T} gy) we already computed:
    //   grad_L_{ij} = -gd_i * y_j    (for j <= i)

    const float gL00 = -gd0 * y0;
    const float gL10 = -gd1 * y0;
    const float gL11 = -gd1 * y1;
    const float gL20 = -gd2 * y0;
    const float gL21 = -gd2 * y1;
    const float gL22 = -gd2 * y2;

    // Accumulate across N (row-major 3x3)
    float* gLk = &grad_L[k * 9];
    atomicAdd(&gLk[0], gL00);                    // [0,0]
    atomicAdd(&gLk[3], gL10);                    // [1,0]
    atomicAdd(&gLk[4], gL11);                    // [1,1]
    atomicAdd(&gLk[6], gL20);                    // [2,0]
    atomicAdd(&gLk[7], gL21);                    // [2,1]
    atomicAdd(&gLk[8], gL22);                    // [2,2]
}


// ============================================================
// C++ interface
// ============================================================
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
    auto grad_L = torch::zeros({K, 9}, x.options());  // (K, 3, 3) flattened

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
        grad_L.data_ptr<float>(),
        N, K
    );

    return {grad_x, grad_means, grad_L.reshape({K, 3, 3}), grad_amplitudes};
}


// ============================================================
// Forward + analytical field gradient kernel
// ============================================================
// Computes both f(x) and ∇_x f(x) in a single pass.
//
// f(x) = Σ_k a_k exp(-0.5 y_k^T y_k)  where  L_k y_k = x - μ_k
// ∇_x f(x) = Σ_k -v_k * L_k^{-T} y_k  =  Σ_k -v_k * Σ_k^{-1}(x - μ_k)
//
// Thread-per-point (not per n*k) — loops over all K Gaussians per point.
// This avoids atomicAdd for accumulation and is efficient for moderate K.
__global__ void forward_with_grad_kernel(
    const float* __restrict__ x,           // (N, 3)
    const float* __restrict__ means,       // (K, 3)
    const float* __restrict__ L_chol,      // (K, 3, 3)
    const float* __restrict__ amplitudes,  // (K,)
    float* __restrict__ output,            // (N,)
    float* __restrict__ field_grad,        // (N, 3)
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float val = 0.0f, gx = 0.0f, gy = 0.0f, gz = 0.0f;

    for (int k = 0; k < K; k++) {
        const float d0 = x[n*3+0] - means[k*3+0];
        const float d1 = x[n*3+1] - means[k*3+1];
        const float d2 = x[n*3+2] - means[k*3+2];

        const float* L = &L_chol[k * 9];
        const float L00 = L[0], L10 = L[3], L11 = L[4];
        const float L20 = L[6], L21 = L[7], L22 = L[8];

        // Forward substitution: y = L^{-1} d
        const float y0 = d0 / L00;
        const float y1 = (d1 - L10*y0) / L11;
        const float y2 = (d2 - L20*y0 - L21*y1) / L22;

        const float mahal = y0*y0 + y1*y1 + y2*y2;
        const float v = amplitudes[k] * expf(-0.5f * mahal);
        val += v;

        // Backward substitution: s = L^{-T} y = Σ^{-1}(x - μ)
        const float s2 = y2 / L22;
        const float s1 = (y1 - L21*s2) / L11;
        const float s0 = (y0 - L10*s1 - L20*s2) / L00;

        // ∇_x v_k = -v * s
        gx += -v * s0;
        gy += -v * s1;
        gz += -v * s2;
    }

    output[n] = val;
    field_grad[n*3+0] = gx;
    field_grad[n*3+1] = gy;
    field_grad[n*3+2] = gz;
}


// ============================================================
// Backward for analytical field gradient supervision
// ============================================================
// Given upstream gradient g (N,3) from L1 loss on field gradient:
//   L = Σ_n Σ_j |∇f_j(x_n) - ∂v/∂x_j(x_n)|
//
// Per (n,k): contribution to ∇f_j is c_{j} = -v_nk * s_{j}
// where v_nk = a_k exp(-0.5 m), s = L^{-T}y = Σ^{-1}(x-μ)
//
// ∂c_j/∂a_k = (v/a) * (-s_j)
// ∂c_j/∂μ_i = -[∂v/∂μ_i * s_j + v * ∂s_j/∂μ_i]
//           = -[v*s_i*s_j - v*Σ^{-1}_{j,i}]   (since ∂v/∂μ=v*s, ∂s/∂μ=-Σ^{-1})
//           = v*(Σ^{-1}_{j,i} - s_i*s_j)
// ∂c_j/∂L: needs chain through y→s→c
__global__ void analytical_grad_supervision_backward_kernel(
    const float* __restrict__ grad_out,     // (N, 3) upstream gradient
    const float* __restrict__ x,            // (N, 3)
    const float* __restrict__ means,        // (K, 3)
    const float* __restrict__ L_chol,       // (K, 3, 3)
    const float* __restrict__ amplitudes,   // (K,)
    float* __restrict__ grad_means,         // (K, 3)
    float* __restrict__ grad_L,             // (K, 9)
    float* __restrict__ grad_amplitudes,    // (K,)
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    const float amp = amplitudes[k];

    const float d0 = x[n*3+0] - means[k*3+0];
    const float d1 = x[n*3+1] - means[k*3+1];
    const float d2 = x[n*3+2] - means[k*3+2];

    const float* L = &L_chol[k * 9];
    const float L00 = L[0], L10 = L[3], L11 = L[4];
    const float L20 = L[6], L21 = L[7], L22 = L[8];

    // Forward substitution: y = L^{-1} d
    const float y0 = d0 / L00;
    const float y1 = (d1 - L10*y0) / L11;
    const float y2 = (d2 - L20*y0 - L21*y1) / L22;

    const float mahal = y0*y0 + y1*y1 + y2*y2;
    const float v = amp * expf(-0.5f * mahal);

    // Backward substitution: s = L^{-T} y
    const float s2 = y2 / L22;
    const float s1 = (y1 - L21*s2) / L11;
    const float s0 = (y0 - L10*s1 - L20*s2) / L00;

    // Upstream gradient for this point
    const float g0 = grad_out[n*3+0];
    const float g1 = grad_out[n*3+1];
    const float g2 = grad_out[n*3+2];

    // g · s  (dot product of upstream grad and s)
    const float gs = g0*s0 + g1*s1 + g2*s2;

    // ---- grad_amplitudes ----
    // c_j = -v*s_j, ∂c_j/∂a = -(v/a)*s_j
    // ∂L/∂a = Σ_j g_j * ∂c_j/∂a = -(v/a) * (g·s)
    if (amp > 1e-12f) {
        atomicAdd(&grad_amplitudes[k], -gs * v / amp);
    }

    // ---- grad_means ----
    // ∂c_j/∂μ_i = v * (Σ^{-1}_{j,i} - s_i*s_j)
    // ∂L/∂μ_i = Σ_j g_j * v * (Σ^{-1}_{j,i} - s_i*s_j)
    //         = v * (Σ^{-1,T} g - s * (g·s))_i
    //         = v * ((Σ^{-1} g)_i - s_i * gs)
    //
    // Σ^{-1} g = L^{-T} L^{-1} g
    // First: L^{-1} g (forward sub on g)
    const float f0 = g0 / L00;
    const float f1 = (g1 - L10*f0) / L11;
    const float f2 = (g2 - L20*f0 - L21*f1) / L22;
    // Then: L^{-T} f (backward sub)
    const float q2 = f2 / L22;
    const float q1 = (f1 - L21*q2) / L11;
    const float q0 = (f0 - L10*q1 - L20*q2) / L00;
    // Σ^{-1} g = (q0, q1, q2)

    const float gm0 = v * (q0 - s0 * gs);
    const float gm1 = v * (q1 - s1 * gs);
    const float gm2 = v * (q2 - s2 * gs);

    atomicAdd(&grad_means[k*3+0], gm0);
    atomicAdd(&grad_means[k*3+1], gm1);
    atomicAdd(&grad_means[k*3+2], gm2);

    // ---- grad_L_chol ----
    // c_j = -v * s_j where v = a*exp(-0.5*m), s = L^{-T}y, y = L^{-1}d
    // ∂c_j/∂L_{pq} involves:
    //   (a) ∂v/∂L_{pq} = v * (-0.5) * ∂m/∂L_{pq}
    //       ∂m/∂L_{pq} = 2 * y_p * ∂y_p/∂L_{pq}  (chain through y)
    //       Using the standard result: ∂y/∂L_{pq} ~ -gd_p * y_q  
    //       where gd = L^{-T}(∂m/∂y) but simplified for Cholesky
    //   (b) ∂s_j/∂L_{pq} involves differentiating L^{-T}y through both L^{-T} and y
    //
    // Rather than derive all terms analytically, use the compact form:
    // From c = -v*s, and using gd (the backward sub of gy through L^T):
    //   grad_L from the value part: like in the main backward kernel
    //   grad_L from the gradient part: from differentiating s w.r.t. L
    //
    // Total gradient of L_{pq} from the field gradient contribution:
    // This uses the "double backward substitution" approach.
    //
    // Part 1: gradient through v (same structure as main backward)
    // ∂(Σ_j g_j * c_j)/∂v = -gs (already computed)
    // ∂v/∂mahal = v * (-0.5)
    // ∂mahal/∂y_i = 2*y_i
    // Chain: grad_y_from_v_i = -gs * v * (-0.5) * 2 * y_i = gs * v * y_i
    const float gvy0 = gs * v * y0;
    const float gvy1 = gs * v * y1;
    const float gvy2 = gs * v * y2;

    // gd from v part: L^{-T} gvy
    const float gvd2 = gvy2 / L22;
    const float gvd1 = (gvy1 - L21*gvd2) / L11;
    const float gvd0 = (gvy0 - L10*gvd1 - L20*gvd2) / L00;

    // grad_L from v part: -gvd_i * y_j (same pattern as main backward)
    float gL00 = -gvd0 * y0;
    float gL10 = -gvd1 * y0;
    float gL11 = -gvd1 * y1;
    float gL20 = -gvd2 * y0;
    float gL21 = -gvd2 * y1;
    float gL22 = -gvd2 * y2;

    // Part 2: gradient through s (the L^{-T}y part)
    // c_j = -v * s_j, upstream for s_j is -v * g_j
    // s = L^{-T} y, so we need ∂s/∂L and ∂s/∂y (which chains through ∂y/∂L)
    //
    // For s = L^{-T} y:
    //   s2 = y2/L22
    //   s1 = (y1 - L21*s2)/L11
    //   s0 = (y0 - L10*s1 - L20*s2)/L00
    //
    // Upstream: gs_j = -v * g_j (grad of loss w.r.t. s_j)
    const float gs0 = -v * g0;
    const float gs1 = -v * g1;
    const float gs2 = -v * g2;

    // Backward through s = L^{-T} y to get grad_y_from_s and grad_L_from_s
    // ∂s0/∂L00 = -s0/L00
    // ∂s0/∂L10 = -(-s1)/L00 = s1/L00  ... wait, let me be careful
    // s0 = (y0 - L10*s1 - L20*s2)/L00
    // ∂s0/∂L00 = -(y0 - L10*s1 - L20*s2)/L00² = -s0/L00
    // ∂s0/∂L10 = -s1/L00
    // ∂s0/∂L20 = -s2/L00
    // s1 = (y1 - L21*s2)/L11
    // ∂s1/∂L11 = -s1/L11
    // ∂s1/∂L21 = -s2/L11
    // s2 = y2/L22
    // ∂s2/∂L22 = -s2/L22
    //
    // Also need grad_y from s (to chain with grad_L from y):
    // ∂s2/∂y2 = 1/L22
    // ∂s1/∂y1 = 1/L11
    // ∂s1/∂y2 = -L21/(L11*L22)  (through s2)
    // ∂s0/∂y0 = 1/L00
    // ∂s0/∂y1 = -L10/(L00*L11)  (through s1)
    // ... this is essentially L^{-1} applied to gs (another forward sub)
    //
    // grad_y_from_s = L^{-1} gs  (forward substitution on gs)
    const float gsy0 = gs0 / L00;
    const float gsy1 = (gs1 - L10*gsy0) / L11;
    const float gsy2 = (gs2 - L20*gsy0 - L21*gsy1) / L22;

    // grad_L from s (direct differentiation): -gs_i * s_j / L_ii equivalent
    // Using the pattern: grad_L_{ij} from backward sub = -gsd_i * s_j
    // where gsd = forward_sub(gs) ... let me use the same pattern as main backward
    // For backward substitution s = L^{-T} y, differentiating w.r.t. L:
    //   grad_L_{ij} = -gbs_i * s_j   where gbs = L^{-1} gs (forward sub)
    // Wait, this follows the same logic: for Ly = d, grad_L_{ij} = -gd_i * y_j
    // For L^T s = y (transpose system), differentiating w.r.t. L^T_{ij}:
    //   grad_L^T_{ij} = -gds_i * s_j  where gds = L^{-1} gs
    // Since L^T_{ij} = L_{ji}, we need to transpose the indices
    // So grad_L_{ji} from s = -gsy_i * s_j
    // i.e. grad_L_{ij} from s = -gsy_j * s_i
    gL00 += -gsy0 * s0;
    gL10 += -gsy0 * s1;  // grad_L[1,0] = -gsy_0 * s_1
    gL11 += -gsy1 * s1;
    gL20 += -gsy0 * s2;  // grad_L[2,0] = -gsy_0 * s_2
    gL21 += -gsy1 * s2;  // grad_L[2,1] = -gsy_1 * s_2
    gL22 += -gsy2 * s2;

    // Now grad_y_from_s needs to chain through y = L^{-1} d to get more grad_L
    // grad_L from y (given upstream gsy): same pattern as main backward
    // gyd = L^{-T} gsy (backward sub)
    const float gyd2 = gsy2 / L22;
    const float gyd1 = (gsy1 - L21*gyd2) / L11;
    const float gyd0 = (gsy0 - L10*gyd1 - L20*gyd2) / L00;

    gL00 += -gyd0 * y0;
    gL10 += -gyd1 * y0;
    gL11 += -gyd1 * y1;
    gL20 += -gyd2 * y0;
    gL21 += -gyd2 * y1;
    gL22 += -gyd2 * y2;

    // Accumulate
    float* gLk = &grad_L[k * 9];
    atomicAdd(&gLk[0], gL00);
    atomicAdd(&gLk[3], gL10);
    atomicAdd(&gLk[4], gL11);
    atomicAdd(&gLk[6], gL20);
    atomicAdd(&gLk[7], gL21);
    atomicAdd(&gLk[8], gL22);
}


std::vector<torch::Tensor> forward_with_field_grad_cuda(
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    // Ensure contiguous row-major layout (cholesky can return column-major)
    x = x.contiguous();
    means = means.contiguous();
    L_chol = L_chol.contiguous();
    amplitudes = amplitudes.contiguous();

    const int N = x.size(0);
    const int K = means.size(0);

    auto output = torch::zeros({N}, x.options());
    auto field_grad = torch::zeros({N, 3}, x.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    forward_with_grad_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        output.data_ptr<float>(),
        field_grad.data_ptr<float>(),
        N, K
    );

    return {output, field_grad};
}


std::vector<torch::Tensor> analytical_grad_supervision_backward_cuda(
    torch::Tensor grad_out,     // (N, 3) upstream gradient
    torch::Tensor x,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    // Ensure contiguous row-major layout
    grad_out = grad_out.contiguous();
    x = x.contiguous();
    means = means.contiguous();
    L_chol = L_chol.contiguous();
    amplitudes = amplitudes.contiguous();

    const int N = x.size(0);
    const int K = means.size(0);

    auto grad_means = torch::zeros_like(means);
    auto grad_amplitudes = torch::zeros_like(amplitudes);
    auto grad_L = torch::zeros({K, 9}, means.options());

    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;

    analytical_grad_supervision_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        x.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_L.data_ptr<float>(),
        grad_amplitudes.data_ptr<float>(),
        N, K
    );

    return {grad_means, grad_L.reshape({K, 3, 3}), grad_amplitudes};
}


// ============================================================
// Fused gradient supervision BACKWARD kernel
// ============================================================
// Computes gradients of the gradient supervision loss w.r.t.
// means, L_chol, and amplitudes in a single fused kernel.
//
// The gradient supervision loss per point n is:
//   L_n = |Δpred_x - Δgt_x| + |Δpred_y - Δgt_y| + |Δpred_z - Δgt_z|
// where Δpred_x = f(x_dx) - f(x_c), Δgt_x = v_dx - v_c, etc.
//
// Each thread handles one (n, k) pair, computing the contribution of
// Gaussian k to the gradient at point n, and accumulates gradients
// via atomicAdd.
__global__ void gradient_supervision_backward_kernel(
    const float* __restrict__ grad_out,     // (N,) upstream gradient (from .mean())
    const float* __restrict__ x_center,     // (N, 3)
    const float* __restrict__ x_dx,         // (N, 3)
    const float* __restrict__ x_dy,         // (N, 3)
    const float* __restrict__ x_dz,         // (N, 3)
    const float* __restrict__ v_center,     // (N,)
    const float* __restrict__ v_dx,         // (N,)
    const float* __restrict__ v_dy,         // (N,)
    const float* __restrict__ v_dz,         // (N,)
    const float* __restrict__ means,        // (K, 3)
    const float* __restrict__ L_chol,       // (K, 3, 3)
    const float* __restrict__ amplitudes,   // (K,)
    // We need the sum predictions to compute signs of the L1 loss
    const float* __restrict__ pred_sums,    // (N, 4) = [pred_c, pred_dx, pred_dy, pred_dz]
    float* __restrict__ grad_means,         // (K, 3)
    float* __restrict__ grad_L,             // (K, 9)
    float* __restrict__ grad_amplitudes,    // (K,)
    const int N,
    const int K
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;

    const int n = idx / K;
    const int k = idx % K;

    const float go = grad_out[n];
    const float amp = amplitudes[k];

    const float* L = &L_chol[k * 9];
    const float L00 = L[0], L10 = L[3], L11 = L[4];
    const float L20 = L[6], L21 = L[7], L22 = L[8];

    // Evaluate this Gaussian k at all 4 points
    // Center
    float dc0 = x_center[n*3+0] - means[k*3+0];
    float dc1 = x_center[n*3+1] - means[k*3+1];
    float dc2 = x_center[n*3+2] - means[k*3+2];
    float yc0 = dc0 / L00;
    float yc1 = (dc1 - L10*yc0) / L11;
    float yc2 = (dc2 - L20*yc0 - L21*yc1) / L22;
    float mc = yc0*yc0 + yc1*yc1 + yc2*yc2;
    float vc = amp * expf(-0.5f * mc);

    // x+dx
    float ddx0 = x_dx[n*3+0] - means[k*3+0];
    float ddx1 = x_dx[n*3+1] - means[k*3+1];
    float ddx2 = x_dx[n*3+2] - means[k*3+2];
    float ydx0 = ddx0 / L00;
    float ydx1 = (ddx1 - L10*ydx0) / L11;
    float ydx2 = (ddx2 - L20*ydx0 - L21*ydx1) / L22;
    float mdx = ydx0*ydx0 + ydx1*ydx1 + ydx2*ydx2;
    float vdx = amp * expf(-0.5f * mdx);

    // y+dy
    float ddy0 = x_dy[n*3+0] - means[k*3+0];
    float ddy1 = x_dy[n*3+1] - means[k*3+1];
    float ddy2 = x_dy[n*3+2] - means[k*3+2];
    float ydy0 = ddy0 / L00;
    float ydy1 = (ddy1 - L10*ydy0) / L11;
    float ydy2 = (ddy2 - L20*ydy0 - L21*ydy1) / L22;
    float mdy = ydy0*ydy0 + ydy1*ydy1 + ydy2*ydy2;
    float vdy = amp * expf(-0.5f * mdy);

    // z+dz
    float ddz0 = x_dz[n*3+0] - means[k*3+0];
    float ddz1 = x_dz[n*3+1] - means[k*3+1];
    float ddz2 = x_dz[n*3+2] - means[k*3+2];
    float ydz0 = ddz0 / L00;
    float ydz1 = (ddz1 - L10*ydz0) / L11;
    float ydz2 = (ddz2 - L20*ydz0 - L21*ydz1) / L22;
    float mdz = ydz0*ydz0 + ydz1*ydz1 + ydz2*ydz2;
    float vdz = amp * expf(-0.5f * mdz);

    // Get the sum predictions (all Gaussians at this point)
    float pred_c  = pred_sums[n*4 + 0];
    float pred_dx_sum = pred_sums[n*4 + 1];
    float pred_dy_sum = pred_sums[n*4 + 2];
    float pred_dz_sum = pred_sums[n*4 + 3];

    // Signed differences
    float diff_x = (pred_dx_sum - pred_c) - (v_dx[n] - v_center[n]);
    float diff_y = (pred_dy_sum - pred_c) - (v_dy[n] - v_center[n]);
    float diff_z = (pred_dz_sum - pred_c) - (v_dz[n] - v_center[n]);

    // Signs for L1 gradient: d|x|/dx = sign(x)
    float sx = (diff_x > 0.0f) ? 1.0f : ((diff_x < 0.0f) ? -1.0f : 0.0f);
    float sy = (diff_y > 0.0f) ? 1.0f : ((diff_y < 0.0f) ? -1.0f : 0.0f);
    float sz = (diff_z > 0.0f) ? 1.0f : ((diff_z < 0.0f) ? -1.0f : 0.0f);

    // grad of L1 loss w.r.t. each per-Gaussian value:
    // L_n = |Σ_k(vdx_k - vc_k) - Δgt_x| + ...
    // ∂L_n/∂vc_k  = -sx - sy - sz  (center appears in all 3 diffs)
    // ∂L_n/∂vdx_k = +sx
    // ∂L_n/∂vdy_k = +sy
    // ∂L_n/∂vdz_k = +sz

    float g_vc  = go * (-sx - sy - sz);
    float g_vdx = go * sx;
    float g_vdy = go * sy;
    float g_vdz = go * sz;

    // Now chain through each evaluation: v = amp * exp(-0.5*m)
    // ∂v/∂amp = exp(-0.5*m) = v/amp
    // ∂v/∂m   = v * (-0.5)
    // ∂m/∂y_i = 2*y_i
    // ∂y/∂d via backward sub: gd = L^{-T} gy
    // ∂d/∂mean = -1, ∂d/∂x = +1

    // Helper: given upstream grad gv for one evaluation,
    // compute and accumulate grads for means, L, amplitudes
    // We'll inline this for each of the 4 evaluations

    // --- grad_amplitudes ---
    float ga = 0.0f;
    if (amp > 1e-12f) {
        ga += g_vc  * vc  / amp;
        ga += g_vdx * vdx / amp;
        ga += g_vdy * vdy / amp;
        ga += g_vdz * vdz / amp;
    }
    atomicAdd(&grad_amplitudes[k], ga);

    // Accumulate grad_means and grad_L from all 4 evaluations
    float gm0 = 0.0f, gm1 = 0.0f, gm2 = 0.0f;
    float gL00_acc = 0.0f, gL10_acc = 0.0f, gL11_acc = 0.0f;
    float gL20_acc = 0.0f, gL21_acc = 0.0f, gL22_acc = 0.0f;

    // Macro for one evaluation's gradient contribution
    #define ACCUM_GRADS(gv, val, y0, y1, y2) \
    { \
        float gm_val = gv * val * (-0.5f); \
        float gy0 = gm_val * 2.0f * y0; \
        float gy1 = gm_val * 2.0f * y1; \
        float gy2 = gm_val * 2.0f * y2; \
        float gd2 = gy2 / L22; \
        float gd1 = (gy1 - L21 * gd2) / L11; \
        float gd0 = (gy0 - L10 * gd1 - L20 * gd2) / L00; \
        gm0 -= gd0; gm1 -= gd1; gm2 -= gd2; \
        gL00_acc += -gd0 * y0; \
        gL10_acc += -gd1 * y0; \
        gL11_acc += -gd1 * y1; \
        gL20_acc += -gd2 * y0; \
        gL21_acc += -gd2 * y1; \
        gL22_acc += -gd2 * y2; \
    }

    ACCUM_GRADS(g_vc,  vc,  yc0,  yc1,  yc2)
    ACCUM_GRADS(g_vdx, vdx, ydx0, ydx1, ydx2)
    ACCUM_GRADS(g_vdy, vdy, ydy0, ydy1, ydy2)
    ACCUM_GRADS(g_vdz, vdz, ydz0, ydz1, ydz2)

    #undef ACCUM_GRADS

    // Accumulate into global arrays
    atomicAdd(&grad_means[k*3+0], gm0);
    atomicAdd(&grad_means[k*3+1], gm1);
    atomicAdd(&grad_means[k*3+2], gm2);

    float* gLk = &grad_L[k * 9];
    atomicAdd(&gLk[0], gL00_acc);
    atomicAdd(&gLk[3], gL10_acc);
    atomicAdd(&gLk[4], gL11_acc);
    atomicAdd(&gLk[6], gL20_acc);
    atomicAdd(&gLk[7], gL21_acc);
    atomicAdd(&gLk[8], gL22_acc);
}


// ============================================================
// Fused gradient supervision kernel (forward)
// ============================================================
// Evaluates field at center + 3 neighbors, computes finite diff gradients
// Input: x_center (N, 3), x_dx (N, 3), x_dy (N, 3), x_dz (N, 3)
//        v_center (N,), v_dx (N,), v_dy (N,), v_dz (N,)
// Output: gradient loss value
__global__ void gradient_supervision_kernel(
    const float* __restrict__ x_center,    // (N, 3)
    const float* __restrict__ x_dx,        // (N, 3) 
    const float* __restrict__ x_dy,        // (N, 3)
    const float* __restrict__ x_dz,        // (N, 3)
    const float* __restrict__ v_center,    // (N,)
    const float* __restrict__ v_dx,        // (N,)
    const float* __restrict__ v_dy,        // (N,)
    const float* __restrict__ v_dz,        // (N,)
    const float* __restrict__ means,       // (K, 3)
    const float* __restrict__ L_chol,      // (K, 3, 3)
    const float* __restrict__ amplitudes,  // (K,)
    float* __restrict__ grad_loss,         // (N,) per-point gradient loss
    float* __restrict__ pred_sums,         // (N, 4) predictions [c, dx, dy, dz]
    const int N,
    const int K
) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    // Evaluate field at 4 points: center, x+dx, y+dy, z+dz
    float pred_c = 0.0f;
    float pred_dx = 0.0f;
    float pred_dy = 0.0f;
    float pred_dz = 0.0f;

    for (int k = 0; k < K; k++) {
        const float* L = &L_chol[k * 9];
        const float L00 = L[0], L10 = L[3], L11 = L[4];
        const float L20 = L[6], L21 = L[7], L22 = L[8];
        const float amp = amplitudes[k];

        // Center point
        float d0 = x_center[n * 3 + 0] - means[k * 3 + 0];
        float d1 = x_center[n * 3 + 1] - means[k * 3 + 1];
        float d2 = x_center[n * 3 + 2] - means[k * 3 + 2];
        float y0 = d0 / L00;
        float y1 = (d1 - L10 * y0) / L11;
        float y2 = (d2 - L20 * y0 - L21 * y1) / L22;
        float mahal = y0*y0 + y1*y1 + y2*y2;
        pred_c += amp * expf(-0.5f * mahal);

        // x + dx
        d0 = x_dx[n * 3 + 0] - means[k * 3 + 0];
        d1 = x_dx[n * 3 + 1] - means[k * 3 + 1];
        d2 = x_dx[n * 3 + 2] - means[k * 3 + 2];
        y0 = d0 / L00;
        y1 = (d1 - L10 * y0) / L11;
        y2 = (d2 - L20 * y0 - L21 * y1) / L22;
        mahal = y0*y0 + y1*y1 + y2*y2;
        pred_dx += amp * expf(-0.5f * mahal);

        // y + dy
        d0 = x_dy[n * 3 + 0] - means[k * 3 + 0];
        d1 = x_dy[n * 3 + 1] - means[k * 3 + 1];
        d2 = x_dy[n * 3 + 2] - means[k * 3 + 2];
        y0 = d0 / L00;
        y1 = (d1 - L10 * y0) / L11;
        y2 = (d2 - L20 * y0 - L21 * y1) / L22;
        mahal = y0*y0 + y1*y1 + y2*y2;
        pred_dy += amp * expf(-0.5f * mahal);

        // z + dz
        d0 = x_dz[n * 3 + 0] - means[k * 3 + 0];
        d1 = x_dz[n * 3 + 1] - means[k * 3 + 1];
        d2 = x_dz[n * 3 + 2] - means[k * 3 + 2];
        y0 = d0 / L00;
        y1 = (d1 - L10 * y0) / L11;
        y2 = (d2 - L20 * y0 - L21 * y1) / L22;
        mahal = y0*y0 + y1*y1 + y2*y2;
        pred_dz += amp * expf(-0.5f * mahal);
    }

    // Compute SIGNED finite difference gradients (preserves edge direction)
    const float grad_pred_x = pred_dx - pred_c;
    const float grad_pred_y = pred_dy - pred_c;
    const float grad_pred_z = pred_dz - pred_c;

    // Ground truth signed gradients
    const float grad_gt_x = v_dx[n] - v_center[n];
    const float grad_gt_y = v_dy[n] - v_center[n];
    const float grad_gt_z = v_dz[n] - v_center[n];

    // L1 loss on signed gradients (preserves rise/fall distinction)
    grad_loss[n] = fabsf(grad_pred_x - grad_gt_x)
                 + fabsf(grad_pred_y - grad_gt_y)
                 + fabsf(grad_pred_z - grad_gt_z);

    // Also store sum predictions for the backward kernel
    pred_sums[n * 4 + 0] = pred_c;
    pred_sums[n * 4 + 1] = pred_dx;
    pred_sums[n * 4 + 2] = pred_dy;
    pred_sums[n * 4 + 3] = pred_dz;
}

std::vector<torch::Tensor> gradient_supervision_cuda(
    torch::Tensor x_center,
    torch::Tensor x_dx,
    torch::Tensor x_dy,
    torch::Tensor x_dz,
    torch::Tensor v_center,
    torch::Tensor v_dx,
    torch::Tensor v_dy,
    torch::Tensor v_dz,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes
) {
    const int N = x_center.size(0);
    const int K = means.size(0);

    auto grad_loss = torch::zeros({N}, x_center.options());
    auto pred_sums = torch::zeros({N, 4}, x_center.options());

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    gradient_supervision_kernel<<<blocks, threads>>>(
        x_center.data_ptr<float>(),
        x_dx.data_ptr<float>(),
        x_dy.data_ptr<float>(),
        x_dz.data_ptr<float>(),
        v_center.data_ptr<float>(),
        v_dx.data_ptr<float>(),
        v_dy.data_ptr<float>(),
        v_dz.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        grad_loss.data_ptr<float>(),
        pred_sums.data_ptr<float>(),
        N, K
    );

    return {grad_loss, pred_sums};
}


std::vector<torch::Tensor> gradient_supervision_backward_cuda(
    torch::Tensor grad_out,       // (N,)
    torch::Tensor x_center,
    torch::Tensor x_dx,
    torch::Tensor x_dy,
    torch::Tensor x_dz,
    torch::Tensor v_center,
    torch::Tensor v_dx,
    torch::Tensor v_dy,
    torch::Tensor v_dz,
    torch::Tensor means,
    torch::Tensor L_chol,
    torch::Tensor amplitudes,
    torch::Tensor pred_sums       // (N, 4)
) {
    const int N = x_center.size(0);
    const int K = means.size(0);

    auto grad_means = torch::zeros_like(means);
    auto grad_amplitudes = torch::zeros_like(amplitudes);
    auto grad_L = torch::zeros({K, 9}, means.options());

    const int threads = 256;
    const int blocks = (N * K + threads - 1) / threads;

    gradient_supervision_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(),
        x_center.data_ptr<float>(),
        x_dx.data_ptr<float>(),
        x_dy.data_ptr<float>(),
        x_dz.data_ptr<float>(),
        v_center.data_ptr<float>(),
        v_dx.data_ptr<float>(),
        v_dy.data_ptr<float>(),
        v_dz.data_ptr<float>(),
        means.data_ptr<float>(),
        L_chol.data_ptr<float>(),
        amplitudes.data_ptr<float>(),
        pred_sums.data_ptr<float>(),
        grad_means.data_ptr<float>(),
        grad_L.data_ptr<float>(),
        grad_amplitudes.data_ptr<float>(),
        N, K
    );

    return {grad_means, grad_L.reshape({K, 3, 3}), grad_amplitudes};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gaussian_eval_forward_cuda, "Gaussian evaluation forward (CUDA)");
    m.def("backward", &gaussian_eval_backward_cuda, "Gaussian evaluation backward (CUDA)");
    m.def("gradient_supervision", &gradient_supervision_cuda, "Fused gradient supervision forward (CUDA)");
    m.def("gradient_supervision_backward", &gradient_supervision_backward_cuda, "Fused gradient supervision backward (CUDA)");
    m.def("forward_with_field_grad", &forward_with_field_grad_cuda, "Forward + analytical field gradient (CUDA)");
    m.def("analytical_grad_backward", &analytical_grad_supervision_backward_cuda, "Analytical gradient supervision backward (CUDA)");
}
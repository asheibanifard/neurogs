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
// OPTIMIZED Backward kernel: per-point with shared memory tiling
// and warp-level reduction for K-parameter gradients.
// ============================================================
//
// Key improvements over the original per-(n,k) kernel:
//  1. One thread per point n, loops over K → eliminates atomicAdd on grad_x
//  2. Gaussian params loaded into shared memory tiles → reduces global reads
//  3. Warp-level __shfl_down_sync reduction → 32× fewer atomicAdds on K-params
//  4. Early skip for negligible Gaussian contributions
//
#define BACKWARD_TILE_K 32

__launch_bounds__(256, 4)
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
    float* __restrict__ grad_L,             // (K, 9)
    const int N,
    const int K
) {
    // Shared memory for Gaussian parameter tiles
    __shared__ float s_means[BACKWARD_TILE_K * 3];   // (TILE, 3)
    __shared__ float s_L[BACKWARD_TILE_K * 6];       // (TILE, 6) lower-tri: L00,L10,L11,L20,L21,L22
    __shared__ float s_amp[BACKWARD_TILE_K];          // (TILE,)

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const bool valid = (n < N);

    // Load point coords into registers
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    if (valid) {
        px = x[n * 3 + 0];
        py = x[n * 3 + 1];
        pz = x[n * 3 + 2];
    }

    // Register accumulators for grad_x — NO atomicAdd needed!
    float gx0 = 0.0f, gx1 = 0.0f, gx2 = 0.0f;

    // Process all K Gaussians in tiles
    for (int tile = 0; tile < K; tile += BACKWARD_TILE_K) {
        const int tile_size = min(BACKWARD_TILE_K, K - tile);

        // Cooperative load: threads < tile_size each load one Gaussian
        if (threadIdx.x < tile_size) {
            const int k = tile + threadIdx.x;
            s_means[threadIdx.x * 3 + 0] = means[k * 3 + 0];
            s_means[threadIdx.x * 3 + 1] = means[k * 3 + 1];
            s_means[threadIdx.x * 3 + 2] = means[k * 3 + 2];
            const float* Lk = &L_chol[k * 9];
            s_L[threadIdx.x * 6 + 0] = Lk[0];  // L00
            s_L[threadIdx.x * 6 + 1] = Lk[3];  // L10
            s_L[threadIdx.x * 6 + 2] = Lk[4];  // L11
            s_L[threadIdx.x * 6 + 3] = Lk[6];  // L20
            s_L[threadIdx.x * 6 + 4] = Lk[7];  // L21
            s_L[threadIdx.x * 6 + 5] = Lk[8];  // L22
            s_amp[threadIdx.x] = amplitudes[k];
        }
        __syncthreads();

        for (int tk = 0; tk < tile_size; tk++) {
            const int k = tile + tk;

            // Read grad_output and val for this (n,k)
            float go = 0.0f, val = 0.0f;
            if (valid) {
                go  = grad_output[n * K + k];
                val = vals[n * K + k];
            }

            // --- Compute per-thread gradient contributions ---
            float lgm0 = 0.0f, lgm1 = 0.0f, lgm2 = 0.0f;
            float lga = 0.0f;
            float lgL00 = 0.0f, lgL10 = 0.0f, lgL11 = 0.0f;
            float lgL20 = 0.0f, lgL21 = 0.0f, lgL22 = 0.0f;
            float lgx0 = 0.0f, lgx1 = 0.0f, lgx2 = 0.0f;

            // Early skip: if contribution is negligible, leave at 0
            if (valid && fabsf(go * val) > 1e-12f) {
                const float amp = s_amp[tk];

                // diff = x_n - mu_k (from shared memory)
                const float d0 = px - s_means[tk * 3 + 0];
                const float d1 = py - s_means[tk * 3 + 1];
                const float d2 = pz - s_means[tk * 3 + 2];

                const float L00 = s_L[tk * 6 + 0];
                const float L10 = s_L[tk * 6 + 1];
                const float L11 = s_L[tk * 6 + 2];
                const float L20 = s_L[tk * 6 + 3];
                const float L21 = s_L[tk * 6 + 4];
                const float L22 = s_L[tk * 6 + 5];

                // Forward sub: y = L^{-1} d
                const float y0 = d0 / L00;
                const float y1 = (d1 - L10 * y0) / L11;
                const float y2 = (d2 - L20 * y0 - L21 * y1) / L22;

                // grad through Mahalanobis
                const float gm = go * val * (-0.5f);
                const float gy0 = gm * 2.0f * y0;
                const float gy1 = gm * 2.0f * y1;
                const float gy2 = gm * 2.0f * y2;

                // Backward sub: gd = L^{-T} gy
                const float gd2 = gy2 / L22;
                const float gd1 = (gy1 - L21 * gd2) / L11;
                const float gd0 = (gy0 - L10 * gd1 - L20 * gd2) / L00;

                // grad_x contribution (accumulated in registers)
                lgx0 = gd0;  lgx1 = gd1;  lgx2 = gd2;

                // grad_means = -gd
                lgm0 = -gd0;  lgm1 = -gd1;  lgm2 = -gd2;

                // grad_amplitudes
                lga = (amp > 1e-12f) ? (go * val / amp) : 0.0f;

                // grad_L_chol: grad_L_{ij} = -gd_i * y_j  (j<=i)
                lgL00 = -gd0 * y0;
                lgL10 = -gd1 * y0;  lgL11 = -gd1 * y1;
                lgL20 = -gd2 * y0;  lgL21 = -gd2 * y1;  lgL22 = -gd2 * y2;
            }

            // Accumulate grad_x directly (no atomic!)
            gx0 += lgx0;  gx1 += lgx1;  gx2 += lgx2;

            // --- Warp-level reduction for K-parameter gradients ---
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lgm0  += __shfl_down_sync(0xffffffff, lgm0,  off);
                lgm1  += __shfl_down_sync(0xffffffff, lgm1,  off);
                lgm2  += __shfl_down_sync(0xffffffff, lgm2,  off);
                lga   += __shfl_down_sync(0xffffffff, lga,   off);
                lgL00 += __shfl_down_sync(0xffffffff, lgL00, off);
                lgL10 += __shfl_down_sync(0xffffffff, lgL10, off);
                lgL11 += __shfl_down_sync(0xffffffff, lgL11, off);
                lgL20 += __shfl_down_sync(0xffffffff, lgL20, off);
                lgL21 += __shfl_down_sync(0xffffffff, lgL21, off);
                lgL22 += __shfl_down_sync(0xffffffff, lgL22, off);
            }

            // Lane 0 of each warp atomicAdds the warp sum (32× fewer atomics)
            if (lane == 0) {
                atomicAdd(&grad_means[k * 3 + 0], lgm0);
                atomicAdd(&grad_means[k * 3 + 1], lgm1);
                atomicAdd(&grad_means[k * 3 + 2], lgm2);
                atomicAdd(&grad_amplitudes[k], lga);
                float* gLk = &grad_L[k * 9];
                atomicAdd(&gLk[0], lgL00);
                atomicAdd(&gLk[3], lgL10);
                atomicAdd(&gLk[4], lgL11);
                atomicAdd(&gLk[6], lgL20);
                atomicAdd(&gLk[7], lgL21);
                atomicAdd(&gLk[8], lgL22);
            }
        }
        __syncthreads();
    }

    // Write grad_x directly (no atomicAdd!)
    if (valid) {
        grad_x[n * 3 + 0] = gx0;
        grad_x[n * 3 + 1] = gx1;
        grad_x[n * 3 + 2] = gx2;
    }
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

    // Per-point launch: N threads (not N*K), each loops over K
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int smem = BACKWARD_TILE_K * (3 + 6 + 1) * sizeof(float);

    gaussian_eval_backward_kernel<<<blocks, threads, smem>>>(
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
// OPTIMIZED Backward for analytical field gradient supervision
// ============================================================
// Per-point with shared memory tiling and warp-level reduction.
// Same math as original but restructured for fewer atomicAdds.
//
__launch_bounds__(256, 4)
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
    __shared__ float s_means[BACKWARD_TILE_K * 3];
    __shared__ float s_L[BACKWARD_TILE_K * 6];
    __shared__ float s_amp[BACKWARD_TILE_K];

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const bool valid = (n < N);

    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f;
    if (valid) {
        px = x[n*3+0];  py = x[n*3+1];  pz = x[n*3+2];
        g0 = grad_out[n*3+0];  g1 = grad_out[n*3+1];  g2 = grad_out[n*3+2];
    }

    for (int tile = 0; tile < K; tile += BACKWARD_TILE_K) {
        const int tile_size = min(BACKWARD_TILE_K, K - tile);

        if (threadIdx.x < tile_size) {
            const int k = tile + threadIdx.x;
            s_means[threadIdx.x * 3 + 0] = means[k * 3 + 0];
            s_means[threadIdx.x * 3 + 1] = means[k * 3 + 1];
            s_means[threadIdx.x * 3 + 2] = means[k * 3 + 2];
            const float* Lk = &L_chol[k * 9];
            s_L[threadIdx.x * 6 + 0] = Lk[0];
            s_L[threadIdx.x * 6 + 1] = Lk[3];
            s_L[threadIdx.x * 6 + 2] = Lk[4];
            s_L[threadIdx.x * 6 + 3] = Lk[6];
            s_L[threadIdx.x * 6 + 4] = Lk[7];
            s_L[threadIdx.x * 6 + 5] = Lk[8];
            s_amp[threadIdx.x] = amplitudes[k];
        }
        __syncthreads();

        for (int tk = 0; tk < tile_size; tk++) {
            const int k = tile + tk;

            float lgm0 = 0, lgm1 = 0, lgm2 = 0;
            float lga = 0;
            float lgL00 = 0, lgL10 = 0, lgL11 = 0;
            float lgL20 = 0, lgL21 = 0, lgL22 = 0;

            if (valid) {
                const float amp = s_amp[tk];
                const float d0 = px - s_means[tk * 3 + 0];
                const float d1 = py - s_means[tk * 3 + 1];
                const float d2 = pz - s_means[tk * 3 + 2];

                const float L00 = s_L[tk * 6 + 0], L10 = s_L[tk * 6 + 1];
                const float L11 = s_L[tk * 6 + 2], L20 = s_L[tk * 6 + 3];
                const float L21 = s_L[tk * 6 + 4], L22 = s_L[tk * 6 + 5];

                const float y0 = d0 / L00;
                const float y1 = (d1 - L10*y0) / L11;
                const float y2 = (d2 - L20*y0 - L21*y1) / L22;

                const float mahal = y0*y0 + y1*y1 + y2*y2;
                const float v = amp * expf(-0.5f * mahal);

                // Skip negligible contributions
                if (fabsf(v) > 1e-12f) {
                    const float s2 = y2 / L22;
                    const float s1 = (y1 - L21*s2) / L11;
                    const float s0 = (y0 - L10*s1 - L20*s2) / L00;

                    const float gs = g0*s0 + g1*s1 + g2*s2;

                    // grad_amplitudes
                    lga = (amp > 1e-12f) ? (-gs * v / amp) : 0.0f;

                    // grad_means: v * (Σ^{-1}g - s*(g·s))
                    const float f0 = g0 / L00;
                    const float f1 = (g1 - L10*f0) / L11;
                    const float f2 = (g2 - L20*f0 - L21*f1) / L22;
                    const float q2 = f2 / L22;
                    const float q1 = (f1 - L21*q2) / L11;
                    const float q0 = (f0 - L10*q1 - L20*q2) / L00;

                    lgm0 = v * (q0 - s0 * gs);
                    lgm1 = v * (q1 - s1 * gs);
                    lgm2 = v * (q2 - s2 * gs);

                    // --- grad_L_chol ---
                    // Part 1: gradient through v
                    const float gvy0 = gs * v * y0;
                    const float gvy1 = gs * v * y1;
                    const float gvy2 = gs * v * y2;
                    const float gvd2 = gvy2 / L22;
                    const float gvd1 = (gvy1 - L21*gvd2) / L11;
                    const float gvd0 = (gvy0 - L10*gvd1 - L20*gvd2) / L00;

                    lgL00 = -gvd0 * y0;
                    lgL10 = -gvd1 * y0;  lgL11 = -gvd1 * y1;
                    lgL20 = -gvd2 * y0;  lgL21 = -gvd2 * y1;  lgL22 = -gvd2 * y2;

                    // Part 2: gradient through s = L^{-T}y
                    const float gs0 = -v * g0, gs1 = -v * g1, gs2 = -v * g2;
                    const float gsy0 = gs0 / L00;
                    const float gsy1 = (gs1 - L10*gsy0) / L11;
                    const float gsy2 = (gs2 - L20*gsy0 - L21*gsy1) / L22;

                    lgL00 += -gsy0 * s0;
                    lgL10 += -gsy0 * s1;  lgL11 += -gsy1 * s1;
                    lgL20 += -gsy0 * s2;  lgL21 += -gsy1 * s2;  lgL22 += -gsy2 * s2;

                    // Part 3: chain grad_y_from_s through y = L^{-1}d
                    const float gyd2 = gsy2 / L22;
                    const float gyd1 = (gsy1 - L21*gyd2) / L11;
                    const float gyd0 = (gsy0 - L10*gyd1 - L20*gyd2) / L00;

                    lgL00 += -gyd0 * y0;
                    lgL10 += -gyd1 * y0;  lgL11 += -gyd1 * y1;
                    lgL20 += -gyd2 * y0;  lgL21 += -gyd2 * y1;  lgL22 += -gyd2 * y2;
                }
            }

            // Warp-level reduction
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lgm0  += __shfl_down_sync(0xffffffff, lgm0,  off);
                lgm1  += __shfl_down_sync(0xffffffff, lgm1,  off);
                lgm2  += __shfl_down_sync(0xffffffff, lgm2,  off);
                lga   += __shfl_down_sync(0xffffffff, lga,   off);
                lgL00 += __shfl_down_sync(0xffffffff, lgL00, off);
                lgL10 += __shfl_down_sync(0xffffffff, lgL10, off);
                lgL11 += __shfl_down_sync(0xffffffff, lgL11, off);
                lgL20 += __shfl_down_sync(0xffffffff, lgL20, off);
                lgL21 += __shfl_down_sync(0xffffffff, lgL21, off);
                lgL22 += __shfl_down_sync(0xffffffff, lgL22, off);
            }

            if (lane == 0) {
                atomicAdd(&grad_means[k * 3 + 0], lgm0);
                atomicAdd(&grad_means[k * 3 + 1], lgm1);
                atomicAdd(&grad_means[k * 3 + 2], lgm2);
                atomicAdd(&grad_amplitudes[k], lga);
                float* gLk = &grad_L[k * 9];
                atomicAdd(&gLk[0], lgL00);
                atomicAdd(&gLk[3], lgL10);
                atomicAdd(&gLk[4], lgL11);
                atomicAdd(&gLk[6], lgL20);
                atomicAdd(&gLk[7], lgL21);
                atomicAdd(&gLk[8], lgL22);
            }
        }
        __syncthreads();
    }
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

    // Per-point launch: N threads, each loops over K
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int smem = BACKWARD_TILE_K * (3 + 6 + 1) * sizeof(float);

    analytical_grad_supervision_backward_kernel<<<blocks, threads, smem>>>(
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
// OPTIMIZED Fused gradient supervision BACKWARD kernel
// ============================================================
// Per-point with shared memory tiling and warp-level reduction.
// Each thread handles one point n, loops over K Gaussians in tiles,
// evaluates the Gaussian at 4 neighbouring points, and accumulates
// gradients for means, L_chol, and amplitudes.
//
__launch_bounds__(256, 4)
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
    const float* __restrict__ pred_sums,    // (N, 4) = [pred_c, pred_dx, pred_dy, pred_dz]
    float* __restrict__ grad_means,         // (K, 3)
    float* __restrict__ grad_L,             // (K, 9)
    float* __restrict__ grad_amplitudes,    // (K,)
    const int N,
    const int K
) {
    __shared__ float s_means[BACKWARD_TILE_K * 3];
    __shared__ float s_L[BACKWARD_TILE_K * 6];
    __shared__ float s_amp[BACKWARD_TILE_K];

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;
    const bool valid = (n < N);

    // Load per-point data into registers
    float cx0 = 0, cx1 = 0, cx2 = 0;
    float dx0 = 0, dx1 = 0, dx2 = 0;
    float dy0 = 0, dy1 = 0, dy2 = 0;
    float dz0 = 0, dz1 = 0, dz2 = 0;
    float go = 0, sx = 0, sy = 0, sz = 0;

    if (valid) {
        cx0 = x_center[n*3+0]; cx1 = x_center[n*3+1]; cx2 = x_center[n*3+2];
        dx0 = x_dx[n*3+0]; dx1 = x_dx[n*3+1]; dx2 = x_dx[n*3+2];
        dy0 = x_dy[n*3+0]; dy1 = x_dy[n*3+1]; dy2 = x_dy[n*3+2];
        dz0 = x_dz[n*3+0]; dz1 = x_dz[n*3+1]; dz2 = x_dz[n*3+2];

        go = grad_out[n];

        // Precompute L1 signs from pred_sums
        float pred_c   = pred_sums[n*4 + 0];
        float pred_dx_s = pred_sums[n*4 + 1];
        float pred_dy_s = pred_sums[n*4 + 2];
        float pred_dz_s = pred_sums[n*4 + 3];

        float diff_x = (pred_dx_s - pred_c) - (v_dx[n] - v_center[n]);
        float diff_y = (pred_dy_s - pred_c) - (v_dy[n] - v_center[n]);
        float diff_z = (pred_dz_s - pred_c) - (v_dz[n] - v_center[n]);

        sx = (diff_x > 0.0f) ? 1.0f : ((diff_x < 0.0f) ? -1.0f : 0.0f);
        sy = (diff_y > 0.0f) ? 1.0f : ((diff_y < 0.0f) ? -1.0f : 0.0f);
        sz = (diff_z > 0.0f) ? 1.0f : ((diff_z < 0.0f) ? -1.0f : 0.0f);
    }

    for (int tile = 0; tile < K; tile += BACKWARD_TILE_K) {
        const int tile_size = min(BACKWARD_TILE_K, K - tile);

        if (threadIdx.x < tile_size) {
            const int k = tile + threadIdx.x;
            s_means[threadIdx.x * 3 + 0] = means[k * 3 + 0];
            s_means[threadIdx.x * 3 + 1] = means[k * 3 + 1];
            s_means[threadIdx.x * 3 + 2] = means[k * 3 + 2];
            const float* Lk = &L_chol[k * 9];
            s_L[threadIdx.x * 6 + 0] = Lk[0];
            s_L[threadIdx.x * 6 + 1] = Lk[3];
            s_L[threadIdx.x * 6 + 2] = Lk[4];
            s_L[threadIdx.x * 6 + 3] = Lk[6];
            s_L[threadIdx.x * 6 + 4] = Lk[7];
            s_L[threadIdx.x * 6 + 5] = Lk[8];
            s_amp[threadIdx.x] = amplitudes[k];
        }
        __syncthreads();

        for (int tk = 0; tk < tile_size; tk++) {
            const int k = tile + tk;

            float lgm0 = 0, lgm1 = 0, lgm2 = 0;
            float lga = 0;
            float lgL00 = 0, lgL10 = 0, lgL11 = 0;
            float lgL20 = 0, lgL21 = 0, lgL22 = 0;

            if (valid) {
                const float amp = s_amp[tk];
                const float mk0 = s_means[tk * 3 + 0];
                const float mk1 = s_means[tk * 3 + 1];
                const float mk2 = s_means[tk * 3 + 2];
                const float L00 = s_L[tk * 6 + 0], L10 = s_L[tk * 6 + 1];
                const float L11 = s_L[tk * 6 + 2], L20 = s_L[tk * 6 + 3];
                const float L21 = s_L[tk * 6 + 4], L22 = s_L[tk * 6 + 5];

                // Evaluate at center
                float y0, y1, y2, m;
                y0 = (cx0 - mk0) / L00;
                y1 = (cx1 - mk1 - L10*y0) / L11;
                y2 = (cx2 - mk2 - L20*y0 - L21*y1) / L22;
                m = y0*y0 + y1*y1 + y2*y2;
                float vc = amp * expf(-0.5f * m);
                float yc0 = y0, yc1 = y1, yc2 = y2;

                // Evaluate at x+dx
                y0 = (dx0 - mk0) / L00;
                y1 = (dx1 - mk1 - L10*y0) / L11;
                y2 = (dx2 - mk2 - L20*y0 - L21*y1) / L22;
                m = y0*y0 + y1*y1 + y2*y2;
                float vdx = amp * expf(-0.5f * m);
                float ydx0 = y0, ydx1 = y1, ydx2 = y2;

                // Evaluate at y+dy
                y0 = (dy0 - mk0) / L00;
                y1 = (dy1 - mk1 - L10*y0) / L11;
                y2 = (dy2 - mk2 - L20*y0 - L21*y1) / L22;
                m = y0*y0 + y1*y1 + y2*y2;
                float vdy = amp * expf(-0.5f * m);
                float ydy0 = y0, ydy1 = y1, ydy2 = y2;

                // Evaluate at z+dz
                y0 = (dz0 - mk0) / L00;
                y1 = (dz1 - mk1 - L10*y0) / L11;
                y2 = (dz2 - mk2 - L20*y0 - L21*y1) / L22;
                m = y0*y0 + y1*y1 + y2*y2;
                float vdz = amp * expf(-0.5f * m);
                float ydz0 = y0, ydz1 = y1, ydz2 = y2;

                // Upstream grads for each evaluation
                float g_vc  = go * (-sx - sy - sz);
                float g_vdx = go * sx;
                float g_vdy = go * sy;
                float g_vdz = go * sz;

                // grad_amplitudes
                if (amp > 1e-12f) {
                    lga = (g_vc * vc + g_vdx * vdx + g_vdy * vdy + g_vdz * vdz) / amp;
                }

                // Macro-inline: accumulate grad_means and grad_L from each evaluation
                #define ACCUM_GRADS_V2(gv, val, _y0, _y1, _y2) \
                { \
                    float gm_val = gv * val * (-0.5f); \
                    float _gy0 = gm_val * 2.0f * _y0; \
                    float _gy1 = gm_val * 2.0f * _y1; \
                    float _gy2 = gm_val * 2.0f * _y2; \
                    float _gd2 = _gy2 / L22; \
                    float _gd1 = (_gy1 - L21 * _gd2) / L11; \
                    float _gd0 = (_gy0 - L10 * _gd1 - L20 * _gd2) / L00; \
                    lgm0 -= _gd0; lgm1 -= _gd1; lgm2 -= _gd2; \
                    lgL00 += -_gd0 * _y0; \
                    lgL10 += -_gd1 * _y0; lgL11 += -_gd1 * _y1; \
                    lgL20 += -_gd2 * _y0; lgL21 += -_gd2 * _y1; lgL22 += -_gd2 * _y2; \
                }

                ACCUM_GRADS_V2(g_vc,  vc,  yc0,  yc1,  yc2)
                ACCUM_GRADS_V2(g_vdx, vdx, ydx0, ydx1, ydx2)
                ACCUM_GRADS_V2(g_vdy, vdy, ydy0, ydy1, ydy2)
                ACCUM_GRADS_V2(g_vdz, vdz, ydz0, ydz1, ydz2)

                #undef ACCUM_GRADS_V2
            }

            // Warp-level reduction
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lgm0  += __shfl_down_sync(0xffffffff, lgm0,  off);
                lgm1  += __shfl_down_sync(0xffffffff, lgm1,  off);
                lgm2  += __shfl_down_sync(0xffffffff, lgm2,  off);
                lga   += __shfl_down_sync(0xffffffff, lga,   off);
                lgL00 += __shfl_down_sync(0xffffffff, lgL00, off);
                lgL10 += __shfl_down_sync(0xffffffff, lgL10, off);
                lgL11 += __shfl_down_sync(0xffffffff, lgL11, off);
                lgL20 += __shfl_down_sync(0xffffffff, lgL20, off);
                lgL21 += __shfl_down_sync(0xffffffff, lgL21, off);
                lgL22 += __shfl_down_sync(0xffffffff, lgL22, off);
            }

            if (lane == 0) {
                atomicAdd(&grad_means[k * 3 + 0], lgm0);
                atomicAdd(&grad_means[k * 3 + 1], lgm1);
                atomicAdd(&grad_means[k * 3 + 2], lgm2);
                atomicAdd(&grad_amplitudes[k], lga);
                float* gLk = &grad_L[k * 9];
                atomicAdd(&gLk[0], lgL00);
                atomicAdd(&gLk[3], lgL10);
                atomicAdd(&gLk[4], lgL11);
                atomicAdd(&gLk[6], lgL20);
                atomicAdd(&gLk[7], lgL21);
                atomicAdd(&gLk[8], lgL22);
            }
        }
        __syncthreads();
    }
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

    // Per-point launch: N threads, each loops over K
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const int smem = BACKWARD_TILE_K * (3 + 6 + 1) * sizeof(float);

    gradient_supervision_backward_kernel<<<blocks, threads, smem>>>(
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
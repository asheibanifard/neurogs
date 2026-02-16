/*
 * CUDA Kernels for NeuroGS Gaussian Splatting
 * ============================================
 * High-performance CUDA kernels for 3D Gaussian mixture volume rendering
 * Optimized for RTX 4090 / A100 architectures
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// ============================================================================
// Utility Functions
// ============================================================================

__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 mat3_mul_vec3(
    const float* R, // 3x3 rotation matrix in row-major
    const float3& v
) {
    return make_float3(
        R[0] * v.x + R[1] * v.y + R[2] * v.z,
        R[3] * v.x + R[4] * v.y + R[5] * v.z,
        R[6] * v.x + R[7] * v.y + R[8] * v.z
    );
}

// Quaternion to rotation matrix (device function)
__device__ __forceinline__ void quat_to_rotmat(
    float qw, float qx, float qy, float qz,
    float* R  // output: 9 floats (row-major)
) {
    float ww = qw * qw, xx = qx * qx, yy = qy * qy, zz = qz * qz;
    float wx = qw * qx, wy = qw * qy, wz = qw * qz;
    float xy = qx * qy, xz = qx * qz, yz = qy * qz;
    
    R[0] = ww + xx - yy - zz;  R[1] = 2.0f * (xy - wz);    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);   R[4] = ww - xx + yy - zz;  R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);   R[7] = 2.0f * (yz + wx);   R[8] = ww - xx - yy + zz;
}


// ============================================================================
// Kernel 1: Dense Gaussian Splatting (Optimized for Shared Memory)
// ============================================================================

/*
 * Evaluates N Gaussians at P query points
 * Uses shared memory tiling for coalesced memory access
 * 
 * Input:
 *   - points: [P, 3] query coordinates
 *   - mu: [N, 3] Gaussian centers
 *   - log_s: [N, 3] log-scale parameters
 *   - q: [N, 4] quaternion rotations (w, x, y, z)
 *   - a: [N] amplitudes
 *   - bias: scalar bias term
 * Output:
 *   - output: [P] rendered values
 */

#define TILE_SIZE 256
#define GAUSSIAN_TILE 32

__global__ void gaussian_splatting_forward_kernel(
    const float* __restrict__ points,      // [P, 3]
    const float* __restrict__ mu,          // [N, 3]
    const float* __restrict__ log_s,       // [N, 3]
    const float* __restrict__ q,           // [N, 4]
    const float* __restrict__ a,           // [N]
    const float bias,
    float* __restrict__ output,            // [P]
    const int P,
    const int N
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p >= P) return;
    
    // Load query point into registers
    const float3 pt = make_float3(points[p * 3], points[p * 3 + 1], points[p * 3 + 2]);
    
    float sum = 0.0f;
    
    // Shared memory for Gaussian parameters (tiled processing)
    __shared__ float s_mu[GAUSSIAN_TILE * 3];
    __shared__ float s_log_s[GAUSSIAN_TILE * 3];
    __shared__ float s_q[GAUSSIAN_TILE * 4];
    __shared__ float s_a[GAUSSIAN_TILE];
    
    const int num_tiles = (N + GAUSSIAN_TILE - 1) / GAUSSIAN_TILE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        const int g_base = tile * GAUSSIAN_TILE;
        const int local_id = threadIdx.x % GAUSSIAN_TILE;
        const int g_idx = g_base + local_id;
        
        // Cooperative loading into shared memory
        if (local_id < GAUSSIAN_TILE && g_idx < N) {
            s_mu[local_id * 3 + 0] = mu[g_idx * 3 + 0];
            s_mu[local_id * 3 + 1] = mu[g_idx * 3 + 1];
            s_mu[local_id * 3 + 2] = mu[g_idx * 3 + 2];
            
            s_log_s[local_id * 3 + 0] = log_s[g_idx * 3 + 0];
            s_log_s[local_id * 3 + 1] = log_s[g_idx * 3 + 1];
            s_log_s[local_id * 3 + 2] = log_s[g_idx * 3 + 2];
            
            s_q[local_id * 4 + 0] = q[g_idx * 4 + 0];
            s_q[local_id * 4 + 1] = q[g_idx * 4 + 1];
            s_q[local_id * 4 + 2] = q[g_idx * 4 + 2];
            s_q[local_id * 4 + 3] = q[g_idx * 4 + 3];
            
            s_a[local_id] = a[g_idx];
        }
        __syncthreads();
        
        // Process all Gaussians in this tile
        const int tile_gaussians = min(GAUSSIAN_TILE, N - g_base);
        
        for (int i = 0; i < tile_gaussians; ++i) {
            // Compute displacement
            float3 dx = make_float3(
                pt.x - s_mu[i * 3 + 0],
                pt.y - s_mu[i * 3 + 1],
                pt.z - s_mu[i * 3 + 2]
            );
            
            // Normalize quaternion
            float qw = s_q[i * 4 + 0];
            float qx = s_q[i * 4 + 1];
            float qy = s_q[i * 4 + 2];
            float qz = s_q[i * 4 + 3];
            float q_norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz + 1e-8f);
            qw /= q_norm; qx /= q_norm; qy /= q_norm; qz /= q_norm;
            
            // Build rotation matrix R (row-major)
            float R[9];
            quat_to_rotmat(qw, qx, qy, qz, R);
            
            // Rotate displacement: w = R @ dx (matches PyTorch einsum convention)
            float3 y = mat3_mul_vec3(R, dx);
            
            // Scale by log_s
            float sx = expf(s_log_s[i * 3 + 0]);
            float sy = expf(s_log_s[i * 3 + 1]);
            float sz = expf(s_log_s[i * 3 + 2]);
            
            // Clamp scales
            sx = fmaxf(1e-4f, fminf(sx, 10.0f));
            sy = fmaxf(1e-4f, fminf(sy, 10.0f));
            sz = fmaxf(1e-4f, fminf(sz, 10.0f));
            
            y.x /= (sx + 1e-8f);
            y.y /= (sy + 1e-8f);
            y.z /= (sz + 1e-8f);
            
            // Gaussian evaluation
            float dist2 = y.x * y.x + y.y * y.y + y.z * y.z;
            float gaussian_val = expf(-0.5f * dist2);
            
            sum += gaussian_val * s_a[i];
        }
        __syncthreads();
    }
    
    output[p] = sum + bias;
}


// ============================================================================
// Kernel 2: Sparse Gaussian Splatting with Culling
// ============================================================================

/*
 * Evaluates only nearby Gaussians using spatial culling
 * Uses a pre-computed list of active Gaussians per point
 */

__global__ void gaussian_splatting_sparse_kernel(
    const float* __restrict__ points,      // [P, 3]
    const int* __restrict__ active_list,   // [total_pairs] Gaussian indices
    const int* __restrict__ point_offsets, // [P+1] cumulative offsets
    const float* __restrict__ mu,          // [N, 3]
    const float* __restrict__ log_s,       // [N, 3]
    const float* __restrict__ q,           // [N, 4]
    const float* __restrict__ a,           // [N]
    const float bias,
    float* __restrict__ output,            // [P]
    const int P
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (p >= P) return;
    
    const float3 pt = make_float3(points[p * 3], points[p * 3 + 1], points[p * 3 + 2]);
    
    const int start = point_offsets[p];
    const int end = point_offsets[p + 1];
    
    float sum = 0.0f;
    
    for (int idx = start; idx < end; ++idx) {
        const int g = active_list[idx];
        
        // Displacement
        float3 dx = make_float3(
            pt.x - mu[g * 3 + 0],
            pt.y - mu[g * 3 + 1],
            pt.z - mu[g * 3 + 2]
        );
        
        // Quaternion normalization
        float qw = q[g * 4 + 0], qx = q[g * 4 + 1];
        float qy = q[g * 4 + 2], qz = q[g * 4 + 3];
        float q_norm = sqrtf(qw * qw + qx * qx + qy * qy + qz * qz + 1e-8f);
        qw /= q_norm; qx /= q_norm; qy /= q_norm; qz /= q_norm;
        
        // Build rotation matrix R (row-major) — NOT transposed
        float R[9];
        quat_to_rotmat(qw, qx, qy, qz, R);
        
        // w = R @ dx (matches PyTorch convention)
        float3 y = mat3_mul_vec3(R, dx);
        
        float sx = fmaxf(1e-4f, fminf(expf(log_s[g * 3 + 0]), 10.0f));
        float sy = fmaxf(1e-4f, fminf(expf(log_s[g * 3 + 1]), 10.0f));
        float sz = fmaxf(1e-4f, fminf(expf(log_s[g * 3 + 2]), 10.0f));
        
        y.x /= (sx + 1e-8f);
        y.y /= (sy + 1e-8f);
        y.z /= (sz + 1e-8f);
        
        float dist2 = y.x * y.x + y.y * y.y + y.z * y.z;
        sum += expf(-0.5f * dist2) * a[g];
    }
    
    output[p] = sum + bias;
}


// ============================================================================
// Kernel 3: Weighted Charbonnier Loss with Reduction
// ============================================================================

/*
 * Computes weighted Charbonnier loss in parallel
 * Uses warp-level reductions for efficiency
 */

__global__ void weighted_charbonnier_loss_kernel(
    const float* __restrict__ pred,
    const float* __restrict__ target,
    const float* __restrict__ weights,
    float* __restrict__ output,
    const int N,
    const float epsilon = 1e-3f
) {
    __shared__ float sdata[256];
    
    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_sum = 0.0f;
    
    if (idx < N) {
        float diff = pred[idx] - target[idx];
        float w = weights[idx];
        local_sum = w * sqrtf(diff * diff + epsilon * epsilon);
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}


// ============================================================================
// Kernel 4: Point Sampling with Importance Weighting
// ============================================================================

/*
 * Samples points from 3D grid with importance weighting
 * Uses parallel prefix sum for efficient sampling
 */

__global__ void importance_sample_kernel(
    const float* __restrict__ importance_map,  // [Z*Y*X]
    const float* __restrict__ rand_values,     // [n_samples]
    int* __restrict__ sampled_indices,         // [n_samples]
    const float* __restrict__ cumsum,          // [Z*Y*X] prefix sum
    const int total_voxels,
    const int n_samples
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n_samples) return;
    
    float target = rand_values[idx];
    
    // Binary search in cumulative sum
    int left = 0, right = total_voxels - 1;
    while (left < right) {
        int mid = (left + right) / 2;
        if (cumsum[mid] < target) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    
    sampled_indices[idx] = left;
}


// ============================================================================
// Kernel 5: Backward Pass for Gaussian Splatting (FULL gradients)
// ============================================================================

/*
 * Computes gradients w.r.t. ALL Gaussian parameters: mu, log_s, q, a, bias
 * Each thread handles one point and atomicAdds into per-Gaussian gradient buffers.
 * 
 * Math (matching PyTorch convention):
 *   out[p] = sum_n a[n] * g(x[p]; mu[n], s[n], q[n]) + bias
 *   g = exp(-0.5 * ||y||^2)
 *   w = R * (x - mu)                        (R @ dx, NOT R^T @ dx)
 *   y = w / s                               (element-wise division by s)
 *   s = clamp(exp(log_s), 1e-4, 10)
 *   R = quat_to_rotmat(normalize(q))
 */

// Helper: R^T @ v  (R stored row-major)
__device__ __forceinline__ float3 mat3_T_mul_vec3(
    const float* R, // 3x3 row-major
    const float3& v
) {
    return make_float3(
        R[0] * v.x + R[3] * v.y + R[6] * v.z,
        R[1] * v.x + R[4] * v.y + R[7] * v.z,
        R[2] * v.x + R[5] * v.y + R[8] * v.z
    );
}

__global__ void gaussian_splatting_backward_kernel(
    const float* __restrict__ grad_output,     // [P]
    const float* __restrict__ points,          // [P, 3]
    const float* __restrict__ mu,              // [N, 3]
    const float* __restrict__ log_s,           // [N, 3]
    const float* __restrict__ q,               // [N, 4]
    const float* __restrict__ a,               // [N]
    float* __restrict__ grad_mu,               // [N, 3]   (zeroed)
    float* __restrict__ grad_log_s,            // [N, 3]   (zeroed)
    float* __restrict__ grad_q,                // [N, 4]   (zeroed)
    float* __restrict__ grad_a,                // [N]      (zeroed)
    float* __restrict__ grad_bias,             // [1]      (zeroed)
    const int P,
    const int N
) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) return;

    const float go = grad_output[p];
    const float3 pt = make_float3(points[p*3], points[p*3+1], points[p*3+2]);

    // bias gradient
    atomicAdd(grad_bias, go);

    for (int n = 0; n < N; ++n) {
        // ---- forward recompute ----
        float3 dx = make_float3(
            pt.x - mu[n*3+0],
            pt.y - mu[n*3+1],
            pt.z - mu[n*3+2]
        );

        // Quaternion normalize
        float qw = q[n*4+0], qx = q[n*4+1], qy = q[n*4+2], qz = q[n*4+3];
        float qnorm = sqrtf(qw*qw + qx*qx + qy*qy + qz*qz + 1e-8f);
        float inv_qnorm = 1.0f / qnorm;
        qw *= inv_qnorm; qx *= inv_qnorm; qy *= inv_qnorm; qz *= inv_qnorm;

        // Build R (row-major)
        float R[9];
        quat_to_rotmat(qw, qx, qy, qz, R);

        // w = R @ dx, y = w / s  (matching PyTorch convention)
        float sx = fmaxf(1e-4f, fminf(expf(log_s[n*3+0]), 10.0f));
        float sy = fmaxf(1e-4f, fminf(expf(log_s[n*3+1]), 10.0f));
        float sz = fmaxf(1e-4f, fminf(expf(log_s[n*3+2]), 10.0f));
        float inv_sx = 1.0f / (sx + 1e-8f);
        float inv_sy = 1.0f / (sy + 1e-8f);
        float inv_sz = 1.0f / (sz + 1e-8f);

        float3 w = mat3_mul_vec3(R, dx);  // R @ dx
        float3 y = make_float3(w.x * inv_sx, w.y * inv_sy, w.z * inv_sz);

        float dist2 = y.x*y.x + y.y*y.y + y.z*y.z;
        float gval = expf(-0.5f * dist2);
        float an = a[n];

        // ---- backward ----
        // d_a[n] += go * gval
        atomicAdd(&grad_a[n], go * gval);

        // d_y = go * a[n] * gval * (-y)   [chain: d(out)/d(dist2) = -0.5*a*g, d(dist2)/d(y) = 2*y]
        float coeff = go * an * gval;  // = go * a * exp(-0.5*dist2)
        float3 d_y = make_float3(-coeff * y.x, -coeff * y.y, -coeff * y.z);

        // d_w from d_y:  y = w / s  =>  d_w = d_y / s
        float3 d_w = make_float3(d_y.x * inv_sx, d_y.y * inv_sy, d_y.z * inv_sz);

        // d_mu: w = R @ dx, dx = x - mu  =>  d(w)/d(mu) = -R
        //   d_mu = -R^T @ d_w
        float3 d_mu_local = mat3_T_mul_vec3(R, d_w);  // R^T @ d_w
        atomicAdd(&grad_mu[n*3+0], -d_mu_local.x);
        atomicAdd(&grad_mu[n*3+1], -d_mu_local.y);
        atomicAdd(&grad_mu[n*3+2], -d_mu_local.z);

        // d_log_s: y_i = w_i / s_i,  d(y_i)/d(s_i) = -w_i / s_i^2
        // d(s_i)/d(log_s_i) = s_i  =>  d_log_s_i = d_y_i * (-w_i / s_i)
        float raw_sx = expf(log_s[n*3+0]);
        float raw_sy = expf(log_s[n*3+1]);
        float raw_sz = expf(log_s[n*3+2]);
        float clamp_mask_x = (raw_sx >= 1e-4f && raw_sx <= 10.0f) ? 1.0f : 0.0f;
        float clamp_mask_y = (raw_sy >= 1e-4f && raw_sy <= 10.0f) ? 1.0f : 0.0f;
        float clamp_mask_z = (raw_sz >= 1e-4f && raw_sz <= 10.0f) ? 1.0f : 0.0f;

        atomicAdd(&grad_log_s[n*3+0], d_y.x * (-w.x * inv_sx) * clamp_mask_x);
        atomicAdd(&grad_log_s[n*3+1], d_y.y * (-w.y * inv_sy) * clamp_mask_y);
        atomicAdd(&grad_log_s[n*3+2], d_y.z * (-w.z * inv_sz) * clamp_mask_z);

        // d_R: w = R @ dx  =>  d_R[r][c] = d_w[r] * dx[c]
        float dR[9];
        dR[0] = d_w.x * dx.x;  dR[1] = d_w.x * dx.y;  dR[2] = d_w.x * dx.z;
        dR[3] = d_w.y * dx.x;  dR[4] = d_w.y * dx.y;  dR[5] = d_w.y * dx.z;
        dR[6] = d_w.z * dx.x;  dR[7] = d_w.z * dx.y;  dR[8] = d_w.z * dx.z;

        // Backprop through quat_to_rotmat (normalized q):
        // R[0]=ww+xx-yy-zz  R[1]=2(xy-wz)   R[2]=2(xz+wy)
        // R[3]=2(xy+wz)     R[4]=ww-xx+yy-zz R[5]=2(yz-wx)
        // R[6]=2(xz-wy)     R[7]=2(yz+wx)    R[8]=ww-xx-yy+zz
        float dqw = 2.0f*(qw*(dR[0]+dR[4]+dR[8]) + qy*dR[2] - qz*dR[1] + qz*dR[3] - qx*dR[5] - qy*dR[6] + qx*dR[7]);
        float dqx = 2.0f*(qx*(dR[0]-dR[4]-dR[8]) + qy*dR[1] + qz*dR[2] + qy*dR[3] - qw*dR[5] + qz*dR[6] + qw*dR[7]);
        float dqy = 2.0f*(qy*(-dR[0]+dR[4]-dR[8]) + qx*dR[1] + qw*dR[2] + qx*dR[3] + qz*dR[5] - qw*dR[6] + qz*dR[7]);
        float dqz = 2.0f*(qz*(-dR[0]-dR[4]+dR[8]) - qw*dR[1] + qx*dR[2] + qw*dR[3] + qy*dR[5] + qx*dR[6] + qy*dR[7]);

        // Backprop through quaternion normalization:
        // q_normalized = q_raw / ||q_raw||
        // d_q_raw = (d_q_norm - q_norm * dot(d_q_norm, q_norm)) / ||q_raw||
        float dot_dq = dqw*qw + dqx*qx + dqy*qy + dqz*qz;
        atomicAdd(&grad_q[n*4+0], (dqw - qw * dot_dq) * inv_qnorm);
        atomicAdd(&grad_q[n*4+1], (dqx - qx * dot_dq) * inv_qnorm);
        atomicAdd(&grad_q[n*4+2], (dqy - qy * dot_dq) * inv_qnorm);
        atomicAdd(&grad_q[n*4+3], (dqz - qz * dot_dq) * inv_qnorm);
    }
}


// ============================================================================
// Host Functions
// ============================================================================

extern "C" {

void launch_gaussian_splatting_forward(
    const float* points,
    const float* mu,
    const float* log_s,
    const float* q,
    const float* a,
    const float bias,
    float* output,
    const int P,
    const int N,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    
    gaussian_splatting_forward_kernel<<<blocks, threads, 0, stream>>>(
        points, mu, log_s, q, a, bias, output, P, N
    );
}

void launch_weighted_charbonnier_loss(
    const float* pred,
    const float* target,
    const float* weights,
    float* output,
    const int N,
    cudaStream_t stream
) {
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    
    // Initialize output to zero
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    
    weighted_charbonnier_loss_kernel<<<blocks, threads, 0, stream>>>(
        pred, target, weights, output, N
    );
}

void launch_gaussian_splatting_backward(
    const float* grad_output,
    const float* points,
    const float* mu,
    const float* log_s,
    const float* q,
    const float* a,
    float* grad_mu,
    float* grad_log_s,
    float* grad_q,
    float* grad_a,
    float* grad_bias,
    const int P,
    const int N,
    cudaStream_t stream
) {
    // Zero all gradient buffers
    cudaMemsetAsync(grad_mu,    0, N * 3 * sizeof(float), stream);
    cudaMemsetAsync(grad_log_s, 0, N * 3 * sizeof(float), stream);
    cudaMemsetAsync(grad_q,     0, N * 4 * sizeof(float), stream);
    cudaMemsetAsync(grad_a,     0, N     * sizeof(float), stream);
    cudaMemsetAsync(grad_bias,  0, 1     * sizeof(float), stream);

    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;

    gaussian_splatting_backward_kernel<<<blocks, threads, 0, stream>>>(
        grad_output, points, mu, log_s, q, a,
        grad_mu, grad_log_s, grad_q, grad_a, grad_bias,
        P, N
    );
}

} // extern "C"

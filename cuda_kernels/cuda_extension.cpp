/*
 * PyTorch C++ Extension Bindings for CUDA Kernels
 * ================================================
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Forward declarations of CUDA kernel launchers
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
    );
    
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
    );
    
    void launch_weighted_charbonnier_loss(
        const float* pred,
        const float* target,
        const float* weights,
        float* output,
        const int N,
        cudaStream_t stream
    );
}

// ============================================================================
// PyTorch Wrapper Functions
// ============================================================================

torch::Tensor gaussian_splatting_forward_cuda(
    torch::Tensor points,      // [P, 3]
    torch::Tensor mu,          // [N, 3]
    torch::Tensor log_s,       // [N, 3]
    torch::Tensor q,           // [N, 4]
    torch::Tensor a,           // [N]
    float bias
) {
    TORCH_CHECK(points.is_cuda(), "points must be a CUDA tensor");
    TORCH_CHECK(mu.is_cuda(), "mu must be a CUDA tensor");
    TORCH_CHECK(log_s.is_cuda(), "log_s must be a CUDA tensor");
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    
    TORCH_CHECK(points.dtype() == torch::kFloat32, "points must be float32");
    TORCH_CHECK(mu.dtype() == torch::kFloat32, "mu must be float32");
    TORCH_CHECK(log_s.dtype() == torch::kFloat32, "log_s must be float32");
    TORCH_CHECK(q.dtype() == torch::kFloat32, "q must be float32");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
    
    const int P = points.size(0);
    const int N = mu.size(0);
    
    TORCH_CHECK(points.size(1) == 3, "points must have shape [P, 3]");
    TORCH_CHECK(mu.size(1) == 3, "mu must have shape [N, 3]");
    TORCH_CHECK(log_s.size(1) == 3, "log_s must have shape [N, 3]");
    TORCH_CHECK(q.size(1) == 4, "q must have shape [N, 4]");
    TORCH_CHECK(a.size(0) == N, "a must have shape [N]");
    
    // Ensure contiguous
    points = points.contiguous();
    mu = mu.contiguous();
    log_s = log_s.contiguous();
    q = q.contiguous();
    a = a.contiguous();
    
    // Allocate output
    auto output = torch::zeros({P}, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(points.device()));
    
    // Launch kernel
    auto stream = c10::cuda::getCurrentCUDAStream(points.device().index());
    launch_gaussian_splatting_forward(
        points.data_ptr<float>(),
        mu.data_ptr<float>(),
        log_s.data_ptr<float>(),
        q.data_ptr<float>(),
        a.data_ptr<float>(),
        bias,
        output.data_ptr<float>(),
        P, N,
        stream.stream()
    );
    
    return output;
}


torch::Tensor weighted_charbonnier_loss_cuda(
    torch::Tensor pred,
    torch::Tensor target,
    torch::Tensor weights
) {
    TORCH_CHECK(pred.is_cuda(), "pred must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");
    
    TORCH_CHECK(pred.dtype() == torch::kFloat32, "pred must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(weights.dtype() == torch::kFloat32, "weights must be float32");
    
    const int N = pred.numel();
    TORCH_CHECK(target.numel() == N, "target must have same size as pred");
    TORCH_CHECK(weights.numel() == N, "weights must have same size as pred");
    
    // Ensure contiguous
    pred = pred.contiguous().view({-1});
    target = target.contiguous().view({-1});
    weights = weights.contiguous().view({-1});
    
    // Allocate output (single scalar)
    auto output = torch::zeros({1}, torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(pred.device()));
    
    auto stream = c10::cuda::getCurrentCUDAStream(pred.device().index());
    launch_weighted_charbonnier_loss(
        pred.data_ptr<float>(),
        target.data_ptr<float>(),
        weights.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        stream.stream()
    );
    
    return output / static_cast<float>(N);
}


// ============================================================================
// CUDA Backward Pass
// ============================================================================

std::vector<torch::Tensor> gaussian_splatting_backward_cuda(
    torch::Tensor grad_output,   // [P]
    torch::Tensor points,        // [P, 3]
    torch::Tensor mu,            // [N, 3]
    torch::Tensor log_s,         // [N, 3]
    torch::Tensor q,             // [N, 4]
    torch::Tensor a              // [N]
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be CUDA");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");

    const int P = points.size(0);
    const int N = mu.size(0);

    grad_output = grad_output.contiguous();
    points = points.contiguous();
    mu = mu.contiguous();
    log_s = log_s.contiguous();
    q = q.contiguous();
    a = a.contiguous();

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(points.device());
    auto grad_mu    = torch::zeros({N, 3}, opts);
    auto grad_log_s = torch::zeros({N, 3}, opts);
    auto grad_q     = torch::zeros({N, 4}, opts);
    auto grad_a     = torch::zeros({N},    opts);
    auto grad_bias  = torch::zeros({1},    opts);

    auto stream = c10::cuda::getCurrentCUDAStream(points.device().index());
    launch_gaussian_splatting_backward(
        grad_output.data_ptr<float>(),
        points.data_ptr<float>(),
        mu.data_ptr<float>(),
        log_s.data_ptr<float>(),
        q.data_ptr<float>(),
        a.data_ptr<float>(),
        grad_mu.data_ptr<float>(),
        grad_log_s.data_ptr<float>(),
        grad_q.data_ptr<float>(),
        grad_a.data_ptr<float>(),
        grad_bias.data_ptr<float>(),
        P, N,
        stream.stream()
    );

    return {grad_mu, grad_log_s, grad_q, grad_a, grad_bias};
}


// ============================================================================
// Autograd Function for Backward Pass
// ============================================================================

class GaussianSplattingFunction : public torch::autograd::Function<GaussianSplattingFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor points,
        torch::Tensor mu,
        torch::Tensor log_s,
        torch::Tensor q,
        torch::Tensor a,
        float bias
    ) {
        ctx->save_for_backward({points, mu, log_s, q, a});
        ctx->saved_data["bias"] = bias;
        
        return gaussian_splatting_forward_cuda(points, mu, log_s, q, a, bias);
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // For now, fall back to PyTorch autograd
        // Full CUDA backward can be implemented for additional speedup
        return {
            torch::Tensor(),  // points
            torch::Tensor(),  // mu
            torch::Tensor(),  // log_s
            torch::Tensor(),  // q
            torch::Tensor(),  // a
            torch::Tensor()   // bias
        };
    }
};

torch::Tensor gaussian_splatting_forward_autograd(
    torch::Tensor points,
    torch::Tensor mu,
    torch::Tensor log_s,
    torch::Tensor q,
    torch::Tensor a,
    float bias
) {
    return GaussianSplattingFunction::apply(points, mu, log_s, q, a, bias);
}


// ============================================================================
// Python Module Definition
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splatting_forward", &gaussian_splatting_forward_cuda,
          "Gaussian splatting forward pass (CUDA)",
          py::arg("points"),
          py::arg("mu"),
          py::arg("log_s"),
          py::arg("q"),
          py::arg("a"),
          py::arg("bias"));
    
    m.def("gaussian_splatting_forward_autograd", &gaussian_splatting_forward_autograd,
          "Gaussian splatting forward with autograd (CUDA)",
          py::arg("points"),
          py::arg("mu"),
          py::arg("log_s"),
          py::arg("q"),
          py::arg("a"),
          py::arg("bias"));
    
    m.def("weighted_charbonnier_loss", &weighted_charbonnier_loss_cuda,
          "Weighted Charbonnier loss (CUDA)",
          py::arg("pred"),
          py::arg("target"),
          py::arg("weights"));
    
    m.def("gaussian_splatting_backward", &gaussian_splatting_backward_cuda,
          "Gaussian splatting backward pass (CUDA)",
          py::arg("grad_output"),
          py::arg("points"),
          py::arg("mu"),
          py::arg("log_s"),
          py::arg("q"),
          py::arg("a"));
}

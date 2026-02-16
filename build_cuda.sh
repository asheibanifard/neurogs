#!/bin/bash
# Build script for NeuroGS CUDA kernels
# ======================================

set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate neurogs
echo "✓ Conda environment 'neurogs' activated"

echo "================================================"
echo "  NeuroGS CUDA Kernel Builder"
echo "================================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found. Please install CUDA Toolkit."
    echo ""
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "✓ Found CUDA version: $CUDA_VERSION"

# Check for PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo "❌ ERROR: PyTorch not found. Please install PyTorch with CUDA support."
    echo ""
    echo "Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

# Check PyTorch CUDA version
TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
echo "✓ PyTorch CUDA version: $TORCH_CUDA"

# Check GPU availability
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
if [ "$GPU_COUNT" -gt 0 ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo "✓ Found $GPU_COUNT GPU(s): $GPU_NAME"
else
    echo "⚠ WARNING: No CUDA GPUs detected. Kernels will compile but won't run."
fi

echo ""
echo "Building CUDA extension..."
echo ""

# Clean previous builds
if [ -d "build" ]; then
    echo "Cleaning previous build..."
    rm -rf build
fi

# Compile
python setup_cuda.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "  ✓ Build successful!"
    echo "================================================"
    echo ""
    echo "Testing import..."
    
    if python -c "import neurogs_cuda; print('✓ CUDA kernels loaded successfully')" 2>/dev/null; then
        echo ""
        echo "Running benchmark..."
        echo ""
        python cuda_ops.py
        echo ""
        echo "================================================"
        echo "  🚀 Ready to accelerate training!"
        echo "================================================"
        echo ""
        echo "Usage:"
        echo "  from cuda_ops import CUDAGaussianMixtureVolume"
        echo "  model = CUDAGaussianMixtureVolume(N0, init_means, init_amp)"
        echo ""
    else
        echo "⚠ WARNING: Build succeeded but import failed."
        echo "Check Python path and library dependencies."
    fi
else
    echo ""
    echo "❌ Build failed. Common issues:"
    echo ""
    echo "1. CUDA version mismatch:"
    echo "   - System CUDA: $CUDA_VERSION"
    echo "   - PyTorch CUDA: $TORCH_CUDA"
    echo "   Make sure they match (at least major version)"
    echo ""
    echo "2. Missing C++ compiler:"
    echo "   - Install: sudo apt-get install g++"
    echo ""
    echo "3. Compute capability not supported:"
    echo "   - Edit setup_cuda.py to add your GPU architecture"
    echo ""
    exit 1
fi

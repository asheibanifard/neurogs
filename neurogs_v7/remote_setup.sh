#!/bin/bash
# Setup script to run on the vast.ai remote server after file transfer

set -e

echo "==== Setting up neurogs_v7 on vast.ai ===="

# Check CUDA availability
echo ""
echo "Checking CUDA..."
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA available: $(nvcc --version | grep release)"
else
    echo "✗ CUDA not found"
    exit 1
fi

# Check Python
echo ""
echo "Checking Python..."
python --version

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -q torch torchvision tqdm pyyaml pillow numpy tifffile || true

# Build CUDA extension (optional)
read -p "Build CUDA extension for 2-3x speedup? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Building CUDA extension..."
    chmod +x build_cuda.sh
    ./build_cuda.sh
    if [ $? -eq 0 ]; then
        echo "✓ CUDA extension built successfully"
    else
        echo "⚠ CUDA build failed, will use PyTorch fallback"
    fi
fi

# Check GPU memory
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

echo ""
echo "==== Setup complete ===="
echo ""
echo "To run training:"
echo "  python neurogs_v7.py"
echo ""
echo "Expected performance:"
echo "  - Volume phase: ~7 steps/sec (24GB GPU)"
echo "  - Hybrid phase: ~2-4 steps/sec"
echo "  - Total time: ~10-15 minutes for 2000 steps"
echo ""
echo "Config (config.yml):"
echo "  - K=5000 Gaussians"
echo "  - 500 volume + 1500 hybrid steps"
echo "  - Mixed precision training"
echo "  - Memory: ~22GB peak usage"

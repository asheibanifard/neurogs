#!/bin/bash
# Build script for CUDA extension

echo "Building custom CUDA extension for faster backward pass..."
echo "This will compile gaussian_eval_cuda.cu"
echo ""

# Check if CUDA is available
if ! command -v nvcc &> /dev/null
then
    echo "ERROR: nvcc (CUDA compiler) not found!"
    echo "Please install CUDA toolkit or check your PATH"
    exit 1
fi

echo "Found CUDA compiler: $(which nvcc)"
echo "CUDA version: $(nvcc --version | grep release)"
echo ""

# Build the extension
python setup_cuda.py install

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Successfully built CUDA extension!"
    echo ""
    echo "The extension will be automatically used for 2-3x faster backward pass."
    echo "Run your training script normally: python neurogs_v7.py"
else
    echo ""
    echo "✗ Build failed!"
    echo ""
    echo "The code will fall back to PyTorch implementation (slower but works)."
    echo "Check error messages above for details."
    exit 1
fi

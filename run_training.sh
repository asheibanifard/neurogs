#!/bin/bash
# Training script with conda environment activation
# =================================================

set -e  # Exit on error

# Activate conda/virtual environment
eval "$(conda shell.bash hook)"
conda activate neurogs
echo "✓ Conda environment 'neurogs' activated"

echo ""
echo "================================================"
echo "  Starting NeuroGS Training"
echo "================================================"
echo ""

# Check and install required dependencies
echo "Checking dependencies..."
python -c "import tifffile" 2>/dev/null || {
    echo "Installing tifffile..."
    pip install tifffile
}

python -c "import tqdm" 2>/dev/null || {
    echo "Installing tqdm..."
    pip install tqdm
}

python -c "import torch" 2>/dev/null || {
    echo "❌ ERROR: PyTorch not installed!"
    echo "Install with: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    exit 1
}

echo "✓ All dependencies satisfied"
echo ""

# Check if .tif file exists
if [ ! -f "10-2900-control-cell-05_cropped_corrected.tif" ]; then
    echo "❌ ERROR: Training data not found!"
    echo "Expected: 10-2900-control-cell-05_cropped_corrected.tif"
    echo ""
    exit 1
fi

# Set library path for CUDA kernels
export LD_LIBRARY_PATH=/venv/neurogs/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}
echo "✓ LD_LIBRARY_PATH set for CUDA kernels"

# Run training
python train_standalone.py "$@"

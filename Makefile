# Makefile for NeuroGS CUDA Extensions
# ======================================

.PHONY: all build install clean test benchmark help train

# Default target
all: build

help:
	@echo "NeuroGS CUDA Extension Build System"
	@echo "===================================="
	@echo ""
	@echo "Targets:"
	@echo "  make build       - Compile CUDA kernels in-place"
	@echo "  make install     - Install CUDA kernels system-wide"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make test        - Run correctness tests"
	@echo "  make benchmark   - Run performance benchmarks"
	@echo "  make train       - Run training script"
	@echo "  make all         - Build (default)"
	@echo ""

build:
	@echo "Building CUDA kernels..."
	@chmod +x build_cuda.sh
	@./build_cuda.sh

install:
	@echo "Installing CUDA kernels..."
	python setup_cuda.py install

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build dist *.egg-info
	@rm -f neurogs_cuda*.so
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Clean complete"

test:
	@echo "Running correctness tests..."
	python -c "import cuda_ops; cuda_ops.benchmark_cuda_kernels()"

benchmark: build
	@echo "Running performance benchmarks..."
	python cuda_ops.py

# Development targets
dev-install:
	@echo "Installing in development mode..."
	python setup_cuda.py develop

format:
	@echo "Formatting CUDA code..."
	@find cuda_kernels -name "*.cu" -o -name "*.cpp" | xargs clang-format -i
	@echo "✓ Format complete"

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@command -v nvcc >/dev/null 2>&1 || { echo "❌ nvcc not found"; exit 1; }
	@python -c "import torch" || { echo "❌ PyTorch not found"; exit 1; }
	@echo "✓ All dependencies satisfied"

# Profile target
profile:
	@echo "Profiling CUDA kernels..."
	nsys profile -o neurogs_profile python cuda_ops.py
	@echo "Profile saved to neurogs_profile.qdrep"
	@echo "View with: nsys-ui neurogs_profile.qdrep"

# Training target
train:
	@echo "Starting training..."
	@chmod +x run_training.sh
	@./run_training.sh

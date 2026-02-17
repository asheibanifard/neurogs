# Transfer Instructions for Vast.ai Server

## Step 1: Add SSH Key to Vast.ai

Your SSH public key is:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFLjH+8viklE5RWfkqjjxHYdnar3KZpWrLuoHu/ONYEa asheibanifard@bournemouth.ac.uk
```

1. Go to https://vast.ai
2. Navigate to Account → SSH Keys
3. Add the above public key

## Step 2: Transfer Files

Once SSH key is added, run:
```bash
scp -P 37319 neurogs_transfer.tar.gz root@ssh3.vast.ai:~/
```

Or transfer individual files:
```bash
scp -P 37319 neurogs_v7.py config.yml gaussian_eval_cuda.cu setup_cuda.py build_cuda.sh root@ssh3.vast.ai:~/
```

## Step 3: Connect and Extract

```bash
ssh -p 37319 root@ssh3.vast.ai -L 8080:localhost:8080
cd ~
tar -xzf neurogs_transfer.tar.gz
```

## Step 4: Setup on Remote Server

```bash
# Build CUDA extension (optional, for 2-3x speedup)
chmod +x build_cuda.sh
./build_cuda.sh

# Install dependencies if needed
pip install torch torchvision tqdm pyyaml

# Run training
python neurogs_v7.py
```

## Files Included

1. **neurogs_v7.py** - Main training script with GPU-optimized volume sampling
2. **config.yml** - Configuration file (K=5000 Gaussians, progressive training)
3. **gaussian_eval_cuda.cu** - Custom CUDA kernel for 2-3x backward pass speedup
4. **setup_cuda.py** - Python build script for CUDA extension
5. **build_cuda.sh** - Shell script to compile CUDA extension

## Note on Data Files

The training data (.tif file) is **not** included in the transfer package (526 MB). 

Options:
- Upload separately via scp: `scp -P 37319 10-2900-control-cell-05_cropped_corrected.tif root@ssh3.vast.ai:~/`
- Generate synthetic data on remote server
- Mount from cloud storage

## Memory Optimization

The code includes adaptive chunk sizing to prevent OOM:
- Automatically detects available GPU memory
- Adjusts batch sizes for MIP rendering
- Frees volume from GPU when entering MIP phase

For K=5000 Gaussians, expect ~22GB GPU memory usage during training.

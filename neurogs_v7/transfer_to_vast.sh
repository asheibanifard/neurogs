#!/bin/bash
# Automated transfer script for vast.ai

set -e

VAST_HOST="root@ssh3.vast.ai"
VAST_PORT="37319"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "==== Transferring files to vast.ai ===="
echo "Host: $VAST_HOST"
echo "Port: $VAST_PORT"
echo ""

# Test connection
echo "Testing SSH connection..."
if ssh -p $VAST_PORT -i $SSH_KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $VAST_HOST "echo 'Connection successful'"; then
    echo "✓ SSH connection works"
else
    echo "✗ SSH connection failed"
    echo ""
    echo "Please add your SSH public key to vast.ai:"
    echo ""
    cat $SSH_KEY.pub
    echo ""
    echo "Go to: https://vast.ai → Account → SSH Keys"
    exit 1
fi

echo ""
echo "Transferring files..."

# Transfer tarball
scp -P $VAST_PORT -i $SSH_KEY -o StrictHostKeyChecking=no \
    neurogs_transfer.tar.gz $VAST_HOST:~/

echo "✓ Transferred neurogs_transfer.tar.gz"

# Optional: transfer data file (large)
if [ -f "10-2900-control-cell-05_cropped_corrected.tif" ]; then
    read -p "Transfer data file (526MB)? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        scp -P $VAST_PORT -i $SSH_KEY -o StrictHostKeyChecking=no \
            10-2900-control-cell-05_cropped_corrected.tif $VAST_HOST:~/
        echo "✓ Transferred data file"
    fi
fi

echo ""
echo "==== Transfer complete ===="
echo ""
echo "To connect and setup:"
echo "  ssh -p $VAST_PORT $VAST_HOST -L 8080:localhost:8080"
echo "  cd ~"
echo "  tar -xzf neurogs_transfer.tar.gz"
echo "  ./build_cuda.sh  # optional"
echo "  python neurogs_v7.py"

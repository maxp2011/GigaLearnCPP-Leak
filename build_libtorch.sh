#!/bin/bash
# Build LibTorch from source with sm_120 (RTX 5090 / Blackwell) support.
# Requires: CUDA 12.8+, git, cmake, ninja, python3, pip.
# Usage: ./build_libtorch.sh [install_dir]
# Output: libtorch in $ROOT/libtorch or $1
#
# On vast.ai with RTX 5090: run this first, then ./build.sh

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${1:-$ROOT/libtorch}"
PYTORCH_SRC="${PYTORCH_SRC:-$ROOT/pytorch-src}"
PYTORCH_REPO="${PYTORCH_REPO:-https://github.com/pytorch/pytorch.git}"
PYTORCH_BRANCH="${PYTORCH_BRANCH:-main}"

# sm_120 = RTX 5090/5080 Blackwell; 12.0 = compute capability for sm_120
# Include common arches + Blackwell for 5090s
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;9.0a;12.0;12.5}"

echo "=== Building LibTorch from source (sm_120/sm_125 for RTX 5090) ==="
echo "  Install dir: $INSTALL_DIR"
echo "  TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "  Source: $PYTORCH_SRC"
echo ""

# Clone PyTorch (full clone with submodules - no --depth to get all submodules)
if [ ! -d "$PYTORCH_SRC" ]; then
    echo "Cloning PyTorch (this may take a while, ~2GB)..."
    git clone --recursive --branch "$PYTORCH_BRANCH" "$PYTORCH_REPO" "$PYTORCH_SRC"
fi
cd "$PYTORCH_SRC"
echo "Updating submodules..."
git submodule sync --recursive
git submodule update --init --recursive
# If nccl is still empty, remove and re-init (common with shallow clones)
if [ ! -d "third_party/nccl/nccl" ] && [ -d "third_party/nccl" ]; then
    echo "Fixing NCCL submodule..."
    rm -rf third_party/nccl
    git submodule update --init third_party/nccl
fi

# Install build deps (Ubuntu)
if command -v apt-get &>/dev/null; then
    echo "Installing cmake, ninja..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq cmake ninja-build python3-dev 2>/dev/null || true
fi

# Configure and build
echo "Configuring (cmake)..."
mkdir -p build
cd build
# CUDA path
CUDA_ROOT="${CUDA_HOME:-}"
[ -z "$CUDA_ROOT" ] && [ -d /usr/local/cuda ] && CUDA_ROOT=/usr/local/cuda
CMAKE_CUDA=""
[ -n "$CUDA_ROOT" ] && CMAKE_CUDA="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT"

cmake .. -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DBUILD_PYTHON=OFF \
    -DBUILD_TEST=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DUSE_NCCL=OFF \
    -DUSE_MPI=OFF \
    -DUSE_NUMA=OFF \
    -DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    $CMAKE_CUDA

echo "Building (30-90 min)..."
ninja -j"$(nproc 2>/dev/null || echo 8)"
ninja install

echo ""
echo "Done! LibTorch installed to: $INSTALL_DIR"
echo "Run: export LIBTORCH_PATH=$INSTALL_DIR && ./build.sh"

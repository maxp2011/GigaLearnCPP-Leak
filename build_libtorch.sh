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

# sm_120 = RTX 5090; compute_125 NOT supported by CUDA 12.9 - force correct list
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;9.0a;12.0"

echo "=== Building LibTorch from source (sm_120 for RTX 5090) ==="
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

# Install build deps (Ubuntu)
if command -v apt-get &>/dev/null; then
    echo "Installing cmake, ninja, python3-dev, libnccl-dev..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq cmake ninja-build python3-dev libnccl-dev libnccl2 2>/dev/null || true
    # If cublas_v2.h missing, CUDA toolkit may be incomplete - try installing
    _cuda_inc="${CUDA_HOME:-/usr/local/cuda}/include"
    if [ ! -f "$_cuda_inc/cublas_v2.h" ] 2>/dev/null; then
        echo "cublas_v2.h not found - install full CUDA toolkit if build fails"
    fi
fi

# NCCL: MUST use system lib - building from source hits compute_125 (unsupported by CUDA 12.9 nvcc)
# Require libnccl-dev; install with: sudo apt install libnccl-dev libnccl2
_has_nccl=false
dpkg -l libnccl-dev 2>/dev/null | grep -q 'ii.*libnccl-dev' && _has_nccl=true
pkg-config --exists nccl 2>/dev/null && _has_nccl=true
[ -f /usr/include/nccl.h ] 2>/dev/null && _has_nccl=true
[ -f /usr/include/x86_64-linux-gnu/nccl.h ] 2>/dev/null && _has_nccl=true
if [ "$_has_nccl" != "true" ]; then
    echo "ERROR: libnccl-dev is required. Install with: sudo apt install libnccl-dev libnccl2"
    echo "Add NVIDIA repo if needed: https://developer.download.nvidia.com/compute/cuda/repos/"
    exit 1
fi
echo "Using system NCCL"

# Configure and build - MUST clear build dir so cmake reconfigures with USE_SYSTEM_NCCL=ON
echo "Configuring (cmake)..."
rm -rf build
echo "  (cleared build dir for fresh config)"
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
    -DUSE_CUFILE=OFF \
    -DUSE_MPI=OFF \
    -DUSE_NUMA=OFF \
    -DCUDAToolkit_ROOT="$CUDA_ROOT" \
    -DTORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    -DUSE_SYSTEM_NCCL=ON \
    -DNCCL_ROOT=/usr \
    $CMAKE_CUDA

echo "Building (30-90 min)..."
ninja -j"$(nproc 2>/dev/null || echo 8)"
ninja install

echo ""
echo "Done! LibTorch installed to: $INSTALL_DIR"
echo "Run: export LIBTORCH_PATH=$INSTALL_DIR && ./build.sh"

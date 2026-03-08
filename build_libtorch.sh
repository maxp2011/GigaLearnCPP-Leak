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

# Patch PyTorch to remove 12.0a (compute_125) - CUDA 12.9 nvcc does not support it
# This fixes NCCL build and all CUDA compilation regardless of USE_SYSTEM_NCCL
SELECT_ARCH="cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
if [ -f "$SELECT_ARCH" ] && ! grep -q "CUDA 12.9 nvcc" "$SELECT_ARCH" 2>/dev/null; then
    echo "Patching select_compute_arch.cmake to remove 12.0a (compute_125)..."
    python3 << 'PYEOF'
import re
path = "cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake"
with open(path) as f:
    content = f.read()
if "CUDA 12.9 nvcc" in content:
    print("Patch already applied")
else:
    # Match: list(APPEND CUDA_ALL_GPU_ARCHITECTURES "12.0a") followed by if(NOT CUDA_VERSION...
    pat = r'( list\(APPEND CUDA_ALL_GPU_ARCHITECTURES "12\.0a"\))\n( if\(NOT CUDA_VERSION VERSION_LESS "13\.0")'
    insert = r'''\1
 # CUDA 12.9 nvcc does not support compute_125 (12.0a)
 if(CUDA_VERSION VERSION_LESS "13.0")
   list(REMOVE_ITEM CUDA_COMMON_GPU_ARCHITECTURES "12.0a")
   list(REMOVE_ITEM CUDA_ALL_GPU_ARCHITECTURES "12.0a")
 endif()
\2'''
    new_content, n = re.subn(pat, insert, content)
    if n > 0:
        with open(path, 'w') as f:
            f.write(new_content)
        print("Patched select_compute_arch.cmake")
    else:
        print("Patch skipped (pattern not found)")
PYEOF
fi

# Patch NCCL build to strip compute_125 - nvcc 12.9 does not support it
# NCCL gets NVCC_GENCODE from torch_cuda_get_nvcc_gencode_flag; we strip 12.0a here
NCCL_CMAKE="cmake/External/nccl.cmake"
NCCL_PATCH="$ROOT/patches/nccl-remove-compute-125.patch"
_nccl_patched=false
if [ -f "$NCCL_CMAKE" ]; then
    if grep -q "CUDA 12.9 nvcc does not support compute_125" "$NCCL_CMAKE" 2>/dev/null; then
        _nccl_patched=true
        echo "nccl.cmake already patched"
    else
        echo "Patching nccl.cmake to strip compute_125 from NVCC_GENCODE..."
        if [ -f "$NCCL_PATCH" ] && git apply --check "$NCCL_PATCH" 2>/dev/null; then
            git apply "$NCCL_PATCH" && _nccl_patched=true && echo "Applied nccl patch via git"
        fi
        if [ "$_nccl_patched" != "true" ]; then
            python3 << 'PYEOF2'
import re
path = "cmake/External/nccl.cmake"
with open(path) as f:
    content = f.read()
if "CUDA 12.9 nvcc does not support compute_125" in content:
    print("nccl.cmake already patched")
    exit(0)
# Match line with optional leading space (PyTorch uses single space)
pat = r'(\s*)string\(REPLACE ";-gencode" " -gencode" NVCC_GENCODE "\$\{NVCC_GENCODE\}"\)'
insert = r'''\1string(REPLACE ";-gencode" " -gencode" NVCC_GENCODE "${NVCC_GENCODE}")
\1# CUDA 12.9 nvcc does not support compute_125 - strip it
\1string(REGEX REPLACE " -gencode=arch=compute_125,code=sm_125" "" NVCC_GENCODE "${NVCC_GENCODE}")
\1string(REGEX REPLACE "-gencode=arch=compute_125,code=sm_125" "" NVCC_GENCODE "${NVCC_GENCODE}")'''
new_content, n = re.subn(pat, insert, content)
if n > 0:
    with open(path, 'w') as f:
        f.write(new_content)
    print("Patched nccl.cmake via Python")
    exit(0)
else:
    print("ERROR: nccl.cmake patch failed - pattern not found")
    exit(1)
PYEOF2
            _nccl_patched=true
        fi
    fi
fi

# Prefer system NCCL when available
NCCL_OPTS=""
_has_nccl=false
dpkg -l libnccl-dev 2>/dev/null | grep -q 'ii.*libnccl-dev' && _has_nccl=true
pkg-config --exists nccl 2>/dev/null && _has_nccl=true
[ -f /usr/include/nccl.h ] 2>/dev/null && _has_nccl=true
[ -f /usr/include/x86_64-linux-gnu/nccl.h ] 2>/dev/null && _has_nccl=true
if [ "$_has_nccl" = "true" ]; then
    echo "Using system NCCL"
    NCCL_OPTS="-DUSE_SYSTEM_NCCL=ON -DNCCL_ROOT=/usr"
else
    echo "No system NCCL - will build from source (patched to remove compute_125)"
fi

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
    $NCCL_OPTS \
    $CMAKE_CUDA

echo "Building (30-90 min)..."
ninja -j"$(nproc 2>/dev/null || echo 8)"
ninja install

echo ""
echo "Done! LibTorch installed to: $INSTALL_DIR"
echo "Run: export LIBTORCH_PATH=$INSTALL_DIR && ./build.sh"

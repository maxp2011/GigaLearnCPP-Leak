#!/bin/bash
# Build GigaLearn on Ubuntu/Linux. Auto-detects libtorch and CUDA.
# Usage: ./build.sh [clean]
# Set LIBTORCH_PATH or CUDA_HOME to override auto-detection.

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# --- LibTorch: prefer project-local, then env, then common locations ---
if [ -n "$LIBTORCH_PATH" ] && [ -d "$LIBTORCH_PATH" ]; then
    LIBTORCH="$LIBTORCH_PATH"
elif [ -d "$ROOT/libtorch" ]; then
    LIBTORCH="$ROOT/libtorch"
elif [ -d "$ROOT/GigaLearnCPP/libtorch" ]; then
    LIBTORCH="$ROOT/GigaLearnCPP/libtorch"
elif [ -d "/usr/local/libtorch" ]; then
    LIBTORCH="/usr/local/libtorch"
else
    echo "LibTorch not found. Either:"
    echo "  1. Extract libtorch into: $ROOT/libtorch"
    echo "  2. Set LIBTORCH_PATH to your libtorch directory"
    echo ""
    echo "Download (CUDA 12.1): wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip"
    echo "  Then: unzip libtorch-*.zip && mv libtorch $ROOT/"
    exit 1
fi
echo "Using LibTorch: $LIBTORCH"

# --- CUDA: optional for GPU build ---
CUDA_ROOT=""
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
    CUDA_ROOT="$CUDA_HOME"
elif [ -n "$CUDA_PATH" ] && [ -d "$CUDA_PATH" ]; then
    CUDA_ROOT="$CUDA_PATH"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_ROOT="/usr/local/cuda"
elif command -v nvcc &>/dev/null; then
    CUDA_ROOT="$(dirname "$(dirname "$(which nvcc)")")"
fi

CMAKE_EXTRA=()
if [ -n "$CUDA_ROOT" ]; then
    echo "Using CUDA: $CUDA_ROOT"
    CMAKE_EXTRA+=(-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT")
else
    echo "CUDA not found - building CPU-only (slower training)"
fi

# --- Clean if requested ---
if [ "${1:-}" = "clean" ]; then
    rm -rf build
    echo "Cleaned build directory"
    shift
fi

# --- Configure ---
mkdir -p build
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH" \
    "${CMAKE_EXTRA[@]}" \
    "$@"

# --- Build ---
cmake --build . --config Release -j"$(nproc 2>/dev/null || echo 4)"
echo ""
echo "Build succeeded. Output: $ROOT/build/"
echo "  - GigaLearnCPP.so (or .dll on Windows)"
echo "  - GigaLearnRLBot (RLBot inference)"
echo "  - GigaLearnBot (standalone training)"

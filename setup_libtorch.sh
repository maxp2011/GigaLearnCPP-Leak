#!/bin/bash
# Download LibTorch into project folder for single-folder deploy.
# Run from project root: ./setup_libtorch.sh [cpu|cu121|cu118]

set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

VARIANT="${1:-cu121}"
case "$VARIANT" in
  cpu)
    URL="https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
    ;;
  cu118)
    URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu118.zip"
    ;;
  cu121)
    URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip"
    ;;
  *)
    echo "Usage: $0 [cpu|cu118|cu121]"
    exit 1
    ;;
esac

if [ -d "$ROOT/libtorch" ]; then
  echo "libtorch already exists at $ROOT/libtorch"
  exit 0
fi

echo "Downloading LibTorch ($VARIANT)..."
wget -q --show-progress "$URL" -O libtorch.zip
echo "Extracting..."
unzip -q libtorch.zip
rm libtorch.zip
echo "Done. libtorch is at $ROOT/libtorch"
echo "Run: ./build.sh"

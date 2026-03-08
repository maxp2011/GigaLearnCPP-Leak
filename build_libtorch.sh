#!/bin/bash
# Build LibTorch from source with sm_120 (RTX 5090 / Blackwell) support.
# Requires: CUDA 12.9, git, cmake, ninja, python3, pip.
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

# sm_120 = RTX 5090; compute_125 (12.5) NOT supported by CUDA 12.9 nvcc - enforce valid list
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;8.9;9.0;9.0a;12.0}"
# Strip 12.5/compute_125 - CUDA 12.9 nvcc does not support it
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//12.5/}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST//;;/;}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST#;}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST%;}"
[ -z "$TORCH_CUDA_ARCH_LIST" ] && TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;9.0a;12.0"
export TORCH_CUDA_ARCH_LIST

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
# NCCL: PyTorch removed it from .gitmodules - clone directly if missing
if [ ! -f "third_party/nccl/Makefile" ] 2>/dev/null; then
    echo "Cloning NCCL (PyTorch no longer includes it as submodule)..."
    rm -rf third_party/nccl 2>/dev/null || true
    mkdir -p third_party
    git clone --depth 1 https://github.com/NVIDIA/nccl.git third_party/nccl
    [ -f "third_party/nccl/Makefile" ] || { echo "ERROR: NCCL clone failed"; exit 1; }
fi

# Install build deps (Ubuntu) - libnccl-dev avoids building NCCL from source
if command -v apt-get &>/dev/null; then
    echo "Installing cmake, ninja, python3-dev, libnccl-dev..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq cmake ninja-build python3-dev 2>/dev/null || true
    sudo apt-get install -y -qq libnccl-dev libnccl2 2>/dev/null || true
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

# Prefer system NCCL when available (avoids NCCL build + compute_125 issues)
NCCL_OPTS=""
_has_nccl=false
dpkg -s libnccl-dev &>/dev/null && _has_nccl=true
pkg-config --exists nccl 2>/dev/null && _has_nccl=true
[ -f /usr/include/nccl.h ] 2>/dev/null && _has_nccl=true
[ -f /usr/include/x86_64-linux-gnu/nccl.h ] 2>/dev/null && _has_nccl=true
[ -f /usr/lib/x86_64-linux-gnu/libnccl.so ] 2>/dev/null && _has_nccl=true
[ -f /usr/lib64/libnccl.so ] 2>/dev/null && _has_nccl=true
if [ "$_has_nccl" = "true" ]; then
    echo "Using system NCCL (libnccl-dev)"
    _nccl_inc="/usr/include"
    [ -f /usr/include/x86_64-linux-gnu/nccl.h ] 2>/dev/null && _nccl_inc="/usr/include/x86_64-linux-gnu"
    _nccl_lib="/usr/lib/x86_64-linux-gnu"
    [ -d /usr/lib64 ] 2>/dev/null && _nccl_lib="/usr/lib64"
    NCCL_OPTS="-DUSE_SYSTEM_NCCL=ON -DNCCL_ROOT=/usr -DNCCL_INCLUDE_DIR=$_nccl_inc -DNCCL_LIB_DIR=$_nccl_lib"
else
    echo "No system NCCL - will build from source (patched to remove compute_125)"
fi
echo "  NCCL_OPTS: ${NCCL_OPTS:-<none>}"

# Configure and build - MUST clear build dir so cmake reconfigures with USE_SYSTEM_NCCL=ON
echo "Configuring (cmake)..."
rm -rf build
echo "  (cleared build dir for fresh config)"

# Ensure third_party/nccl exists - cmake requires it even when USE_SYSTEM_NCCL=ON (PyTorch bug/workaround)
if [ ! -f "third_party/nccl/Makefile" ] 2>/dev/null; then
    echo "Cloning NCCL (required by cmake)..."
    rm -rf third_party/nccl 2>/dev/null || true
    mkdir -p third_party
    git clone --depth 1 https://github.com/NVIDIA/nccl.git third_party/nccl || { echo "ERROR: NCCL clone failed"; exit 1; }
fi
[ -f "third_party/nccl/Makefile" ] || { echo "ERROR: third_party/nccl missing"; exit 1; }

# Patch nccl.cmake: (1) add DOWNLOAD_COMMAND when NCCL source missing, (2) strip compute_125
NCCL_CMAKE="cmake/External/nccl.cmake"
PATCH_FILE="$ROOT/patches/nccl-remove-compute-125.patch"
if [ -f "$NCCL_CMAKE" ]; then
    # Add DOWNLOAD_COMMAND so cmake clones NCCL when third_party/nccl is empty (fixes ExternalProject error)
    if ! grep -q 'NCCL_DOWNLOAD_CMD' "$NCCL_CMAKE" 2>/dev/null; then
        echo "Patching nccl.cmake to add DOWNLOAD_COMMAND for missing NCCL source..."
        python3 << 'PYDOWNLOAD'
path = 'cmake/External/nccl.cmake'
with open(path) as f:
    content = f.read()
# Insert before "set(__NCCL_BUILD_DIR" - add conditional download command
marker = 'set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")'
insert = '''set(__NCCL_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/nccl")
if(NOT EXISTS "${__NCCL_SRC_DIR}/Makefile")
  find_package(Git QUIET REQUIRED)
  set(__NCCL_DOWNLOAD_CMD
    ${CMAKE_COMMAND} -E make_directory ${PROJECT_SOURCE_DIR}/third_party
    COMMAND ${GIT_EXECUTABLE} clone --depth 1 https://github.com/NVIDIA/nccl.git ${__NCCL_SRC_DIR}
  )
else()
  set(__NCCL_DOWNLOAD_CMD ${CMAKE_COMMAND} -E echo "NCCL source exists")
endif()

'''
if marker in content and 'NCCL_DOWNLOAD_CMD' not in content:
    content = content.replace(marker, insert + marker)
    # Insert DOWNLOAD_COMMAND into ExternalProject_Add (before SOURCE_DIR)
    import re
    ep_pat = r'(ExternalProject_Add\(nccl_external\s*\n)(\s*)(SOURCE_DIR \$\{PROJECT_SOURCE_DIR\}/third_party/nccl)'
    content = re.sub(ep_pat, r'\1\2DOWNLOAD_COMMAND ${__NCCL_DOWNLOAD_CMD}\n\2\3', content, count=1)
    with open(path, 'w') as f:
        f.write(content)
    print('  Added DOWNLOAD_COMMAND to nccl.cmake')
else:
    print('  DOWNLOAD_COMMAND patch skipped (already applied or marker not found)')
PYDOWNLOAD
    fi
    # Strip compute_125 from NVCC_GENCODE (CUDA 12.9 nvcc does not support it)
    if ! grep -q 'REGEX REPLACE.*compute_125' "$NCCL_CMAKE" 2>/dev/null; then
        echo "Patching nccl.cmake to strip compute_125 from NVCC_GENCODE..."
        if [ -f "$PATCH_FILE" ] && git apply -p1 --check < "$PATCH_FILE" 2>/dev/null; then
            git apply -p1 < "$PATCH_FILE" && echo "  Applied via git apply"
        else
            python3 << 'PYNCCL'
import sys
path = 'cmake/External/nccl.cmake'
with open(path) as f:
    lines = f.readlines()
out, done = [], False
for line in lines:
    out.append(line)
    if not done and ';-gencode' in line and 'NVCC_GENCODE' in line and 'REGEX' not in line:
        indent = line[:len(line) - len(line.lstrip())]
        out.append(indent + '# CUDA 12.9 nvcc does not support compute_125 - strip it\n')
        out.append(indent + 'string(REGEX REPLACE " -gencode=arch=compute_125,code=sm_125" "" NVCC_GENCODE "${NVCC_GENCODE}")\n')
        out.append(indent + 'string(REGEX REPLACE "-gencode=arch=compute_125,code=sm_125" "" NVCC_GENCODE "${NVCC_GENCODE}")\n')
        done = True
if not done:
    sys.exit('ERROR: nccl.cmake patch failed - target line not found')
with open(path, 'w') as f:
    f.writelines(out)
print('Patched nccl.cmake (Python fallback)')
PYNCCL
        fi
    fi
fi

# Patch NCCL makefiles/common.mk to filter compute_125 at build time
# CUDA 12.9 nvcc does not support compute_125; this is where NVCC_GENCODE is used
NCCL_COMMON_MK="third_party/nccl/makefiles/common.mk"
if [ -f "$NCCL_COMMON_MK" ] && ! grep -q "filter compute_125" "$NCCL_COMMON_MK" 2>/dev/null; then
    echo "Patching NCCL makefiles/common.mk to filter compute_125 from NVCC_GENCODE..."
    # Insert override right after NVCC_GENCODE is set (before $(info NVCC_GENCODE...))
    python3 << 'PYNCCLMK'
path = "third_party/nccl/makefiles/common.mk"
with open(path) as f:
    content = f.read()
# Insert after the last NVCC_GENCODE ?= block, before $(info NVCC_GENCODE...)
marker = "$(info NVCC_GENCODE is ${NVCC_GENCODE})"
patch = '''# CUDA 12.9 nvcc does not support compute_125 - filter at build time
override NVCC_GENCODE := $(shell echo '$(NVCC_GENCODE)' | sed 's/ -gencode=arch=compute_125,code=sm_125//g' | sed 's/-gencode=arch=compute_125,code=sm_125//g')

'''
if marker in content and "filter compute_125" not in content:
    content = content.replace(marker, patch + marker)
    with open(path, "w") as f:
        f.write(content)
    print("  Patched makefiles/common.mk")
else:
    raise SystemExit("ERROR: NCCL common.mk patch failed - marker not found or already patched")
PYNCCLMK
fi

mkdir -p build
cd build
# CUDA 12.9 path (set CUDA_HOME if you have multiple CUDA installs)
CUDA_ROOT="${CUDA_HOME:-}"
[ -z "$CUDA_ROOT" ] && [ -d /usr/local/cuda ] && CUDA_ROOT=/usr/local/cuda
[ -n "$CUDA_ROOT" ] && echo "  CUDA: $CUDA_ROOT ($(nvcc --version 2>/dev/null | grep release | sed 's/.*release //;s/,.*//' || echo 'version unknown'))"
CMAKE_CUDA=""
[ -n "$CUDA_ROOT" ] && CMAKE_CUDA="-DCUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT"

# Put USE_SYSTEM_NCCL first so it's not overridden by defaults
cmake .. -G Ninja \
    $NCCL_OPTS \
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
    $CMAKE_CUDA

# Post-cmake: strip compute_125 from generated build files (belt-and-suspenders; NCCL Makefile patch is primary)
echo "Stripping compute_125 from build files..."
python3 << 'PYSTRIP'
import os, re
patterns = [
    (r'\s*-gencode=arch=compute_125,code=sm_125', ''),  # any whitespace before
    (r' -gencode=arch=compute_125,code=sm_125', ''),
    (r'-gencode=arch=compute_125,code=sm_125', ''),
]
count = 0
for root, _, files in os.walk('.'):
    for n in files:
        if n.endswith(('.o', '.a', '.so', '.dylib')): continue
        p = os.path.join(root, n)
        try:
            with open(p, 'r', errors='ignore') as f:
                s = f.read()
            if 'compute_125' not in s: continue
            orig = s
            for pat, repl in patterns:
                s = re.sub(pat, repl, s)
            if s != orig:
                with open(p, 'w') as f:
                    f.write(s)
                print('  Patched:', p)
                count += 1
        except Exception:
            pass
print('Stripped compute_125 from', count, 'file(s)' if count else '(none found)')
PYSTRIP

echo "Building (30-90 min)..."
ninja -j"$(nproc 2>/dev/null || echo 8)"
ninja install

echo ""
echo "Done! LibTorch installed to: $INSTALL_DIR"
echo "Run: export LIBTORCH_PATH=$INSTALL_DIR && ./build.sh"

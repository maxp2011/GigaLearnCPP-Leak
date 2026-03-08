# Deploy to Ubuntu VPS (Single-Folder Setup)

Everything lives in one folder. Copy the project, add libtorch, and build.

## Quick transfer (one archive)

```bash
# On your Windows machine: zip project (exclude build, libtorch if huge)
# Or use rsync/scp to sync the folder
rsync -avz --exclude build --exclude libtorch GigaLearnCPP-Leak/ user@vps:/home/user/GigaLearnCPP-Leak/
```

## Folder layout

```
GigaLearnCPP-Leak/
├── libtorch/              # Add this: LibTorch (download below)
├── build/                 # Created by build
├── build.sh               # Ubuntu build script
├── build.cmd              # Windows build script
├── CMakeLists.txt
├── GigaLearnCPP/
├── RLBotCPP/
├── src/
└── ...
```

## 1. Copy project to VPS

```bash
# From your machine (or zip and scp)
scp -r GigaLearnCPP-Leak user@vps:/home/user/
```

## 2. Add LibTorch (one-time)

Download LibTorch for Linux + CUDA into the project folder:

```bash
cd /home/user/GigaLearnCPP-Leak

# CUDA 12.1 (recommended for most GPUs)
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu121.zip
unzip libtorch-*.zip
rm libtorch-*.zip

# Or CPU-only (no GPU):
# wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
# unzip libtorch-*.zip && rm libtorch-*.zip
```

LibTorch must end up at `./libtorch` (or `./GigaLearnCPP/libtorch`).

## 3. Install build deps (Ubuntu)

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev
# For CUDA build:
sudo apt install -y nvidia-cuda-toolkit   # or install CUDA from nvidia.com
```

## 4. Build

```bash
cd /home/user/GigaLearnCPP-Leak
chmod +x build.sh
./build.sh
```

Output: `build/libGigaLearnCPP.so`, `build/GigaLearnBot`, `build/GigaLearnRLBot`

## Override paths (optional)

| Env var       | Use when |
|---------------|----------|
| `LIBTORCH_PATH` | LibTorch is elsewhere |
| `CUDA_HOME`     | CUDA is not in `/usr/local/cuda` |

```bash
export LIBTORCH_PATH=/opt/libtorch
export CUDA_HOME=/usr/local/cuda-12.1
./build.sh
```

## Run training

```bash
cd build
./GigaLearnBot   # or your usual launch command
```

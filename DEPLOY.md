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
├── collision_meshes/      # Add this: RocketSim arena meshes (see step 5)
│   └── soccar/*.cmf
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

### RTX 5090 / Blackwell (sm_120) – build LibTorch from source

Pre-built LibTorch does not include sm_120 kernels. On RTX 5090 you'll see "no kernel image is available for execution on the device". Build LibTorch from source:

```bash
cd /workspace/GigaLearnCPP-Leak
# Optional: sudo apt install -y libnccl-dev libnccl2  # avoids NCCL build + compute_125 fix
chmod +x build_libtorch.sh
./build_libtorch.sh   # 30–90 min, needs CUDA 12.9
export LIBTORCH_PATH=$(pwd)/libtorch
./build.sh
```

Requires: CUDA 12.9, git, cmake, ninja. The script clones PyTorch and builds with `TORCH_CUDA_ARCH_LIST` including 12.0 (RTX 5090). If `libnccl-dev` is installed, system NCCL is used (recommended). Otherwise the script patches PyTorch to remove compute_125 from the NCCL build (unsupported by CUDA 12.9 nvcc).

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

## 5. Add collision meshes (required for RocketSim)

RocketSim needs arena collision meshes (`collision_meshes/soccar/*.cmf`). Without them you get "No arena meshes found for gamemode soccar".

**Option A – Clone from GitHub** (if you've uploaded to [maxp2011/Collision_Meshes](https://github.com/maxp2011/Collision_Meshes)):

```bash
cd /workspace/GigaLearnCPP-Leak
chmod +x setup_collision_meshes.sh
./setup_collision_meshes.sh
```

**Option B – Upload from Windows** (one-time, to populate Collision_Meshes repo):

```powershell
# On Windows (PowerShell), from the project folder:
.\upload_collision_meshes.ps1
```

Then use Option A on the server.

**Option C – Copy manually** (zip from Windows, unzip on server):

Place `collision_meshes/soccar/*.cmf` in the project root.

**Option D – Generate with RLArenaCollisionDumper** (Windows + Rocket League):

1. Download [RLArenaCollisionDumper](https://github.com/ZealanL/RLArenaCollisionDumper)
2. Run it with Rocket League installed; it outputs `.cmf` files
3. Put them in `collision_meshes/soccar/` and run `upload_collision_meshes.ps1`

The build copies `collision_meshes` into the build dir if it exists. Or set `COLLISION_MESH_PATH` to override at runtime:

```bash
export COLLISION_MESH_PATH=/path/to/collision_meshes
./GigaLearnBot
```

## Run training

```bash
cd build
./GigaLearnBot   # or your usual launch command
```

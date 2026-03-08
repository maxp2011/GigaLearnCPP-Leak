# CUDA Acceleration Plan — Full Rewrite Scope

## What "Entire Thing in CUDA" Means

| Component | Current | CUDA Rewrite | Feasibility |
|-----------|---------|--------------|-------------|
| **RocketSim** | C++/Bullet physics | Custom CUDA physics engine | **Months** — 10k+ lines, collision meshes, vehicle dynamics |
| **Observation building** | CPU per-player | Batched CUDA kernel | **Days** — math is parallelizable |
| **Policy/Critic inference** | LibTorch (cuDNN/cuBLAS) | Custom fused kernels | **Weeks** — LibTorch already optimized; marginal gains |
| **GAE** | CUDA kernel ✓ | Done | **Done** |
| **PPO Learn** | LibTorch autograd | Custom backward kernels | **Months** — reimplement autograd |
| **Experience buffer** | CPU vectors → GPU | GPU-native storage | **Days** |
| **Reward computation** | CPU per-step | Batched CUDA | **Days** |

**Reality:** LibTorch already runs on CUDA. RocketSim cannot be ported without a full physics rewrite. The highest-impact wins are:
1. **Batch obs building** — eliminate CPU obs + reduce transfer
2. **GPU-native experience** — keep data on device
3. **Overlap** — pipeline collection with consumption

## Implemented

- [x] GAE CUDA kernel (no CPU round-trip)
- [x] Experience kept on GPU during Learn
- [x] **CustomObs CUDA batch kernel** — `CustomObs.cu`, `CustomObsCuda.cpp`
  - Batched observation building for all players across arenas
  - Packed format: 350 floats/arena (ball, pads, timers, 6×34 player slots)
  - One thread per player; in-arena inversion and closest-to-ball computed on GPU
- [x] **InferUnit CUDA path** — when `obsSize==323` and batch layout matches, uses GPU obs build (no CPU round-trip)

## RocketSim on GPU

Porting RocketSim to CUDA would require:
- CUDA-accelerated collision detection (replace Bullet)
- Parallel arena stepping (each arena independent)
- GPU-friendly data structures

Projects like [cuCollide](https://github.com/NVIDIA/cuCollide) exist for collision; vehicle dynamics would need custom work. **Not in scope for this codebase.**

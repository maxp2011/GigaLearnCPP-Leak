# ULTIMATE Collection Steps/Second Guide

Collection throughput = `stepsCollected / collectionTime`. Each "step" does: **Reset** → **Inference** → **StepSecondHalf**. Maximize parallelism and minimize per-step cost.

---

## 🔥 Highest impact (try first)

### 1. **Enable half-precision inference**
FP16 on Tensor Cores = ~2× faster policy forward.
```cpp
cfg.ppo.useHalfPrecision = true;  // in ExampleMain.cpp
```

### 2. **Increase tick skip**
Fewer physics ticks per action = faster.
- Current: `tickSkip = 6` → 6 ticks/step
- Try: `tickSkip = 8` (33% fewer ticks, slightly less control granularity)

### 3. **More games + match threads**
More parallel arenas = more steps per "step".
- `cfg.numGames = 500` or `650` (or more if GPU/CPU allow)
- `cfg.numThreads = cfg.numGames` (threads ≥ arenas for full parallelism)

### 4. **Smaller policy network**
Faster inference at some cost to capacity.
- `cfg.ppo.policy.layerSizes = { 256, 256, 256 };`
- `cfg.ppo.critic.layerSizes = { 256, 256, 256 };`

---

## ⚡ Medium impact

### 5. **Remove / reduce expensive rewards**
Each reward runs per arena. Fewer or simpler rewards = faster `StepSecondHalf`.
- Comment out heavy rewards (complex physics, ball-state checks).
- Use simpler substitutes where possible.

### 6. **Simpler observation**
- Try `StreamlinedObs` instead of `AdvancedObsPadded` if compatible (smaller obs = faster).
- Or a smaller obs size (fewer features) if you can afford it.

### 7. **Disable optional overhead**
- `config.addRewardsToMetrics = false` if you don't need reward breakdowns.
- Pass `nullptr` for `StepCallback` if you don't need step-level metrics.

### 8. **numThreads tuning**
- Too low: arenas wait for threads.
- Too high: thread contention.
- Rule of thumb: `numThreads = numGames` or `numGames + some margin`.

---

## 🚀 Advanced (code changes)

### 9. **CUDA graphs for inference**
Capture the policy forward pass as a CUDA graph and replay it. Removes per-call kernel launch overhead (big win for small batches). Requires changes in `InferUnit` / `PPOLearner`.

### 10. **TorchScript / JIT**
JIT-compile the policy for faster inference:
```cpp
torch::jit::trace(model, example_input);
```

### 11. **Overlap Reset and StepSecondHalf**
Run Reset for step N+1 in parallel with StepSecondHalf for step N. Requires restructuring the collection loop (double-buffering).

### 12. **Reduce obs stat / reward sampling**
- `config.maxObsSamples` lower (or 0 if not needed).
- `config.rewardSampleRandInterval` higher (sample less often).

---

## 📊 Quick config bundle (copy-paste)

```cpp
cfg.tickSkip = 8;
cfg.numGames = 650;
cfg.numThreads = 650;
cfg.ppo.useHalfPrecision = true;
cfg.ppo.policy.layerSizes = { 256, 256, 256 };
cfg.ppo.critic.layerSizes = { 256, 256, 256 };
```

---

## 🎯 Typical bottlenecks

1. **Inference** – GPU policy forward. Fix: half precision, smaller net, CUDA graphs.
2. **Env stepping** – RocketSim physics. Fix: more threads, higher tick skip, fewer/simpler rewards.
3. **Reset** – Arena reset. Fix: more threads (already parallel).
4. **Obs/reward compute** – CPU work in StepSecondHalf. Fix: simpler obs, fewer rewards.

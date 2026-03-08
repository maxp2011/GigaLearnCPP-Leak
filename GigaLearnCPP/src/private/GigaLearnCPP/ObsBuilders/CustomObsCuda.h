#pragma once

#ifdef RG_CUDA_SUPPORT
#include <cuda_runtime.h>
#include <vector>

namespace GGL {

// CustomObs CUDA batch builder — matches CustomObs.h layout exactly.
// Packed arena stride: 9 (ball) + 34*2 (pads) + 34*2 (timers) + 1 (numPlayers) + 6*34 (players) = 350 floats
constexpr int CUSTOM_OBS_CUDA_ARENA_STRIDE = 350;
constexpr int CUSTOM_OBS_CUDA_OBS_SIZE = 323;
constexpr int CUSTOM_OBS_CUDA_MAX_PLAYERS_PER_ARENA = 6;
constexpr int CUSTOM_OBS_CUDA_PLAYER_FEAT_SIZE = 44;

// Pack game states into a flat GPU buffer. Caller allocates d_packed (numArenas * CUSTOM_OBS_CUDA_ARENA_STRIDE floats).
// arenaPlayerStartIdx: length numArenas+1, arenaPlayerStartIdx[i] = first global player index for arena i
void CustomObsCuda_PackArenas(
	const std::vector<const void*>& gameStates,
	const std::vector<int>& arenaPlayerStartIdx,
	float* d_packed,
	cudaStream_t stream = 0
);

// Build observations on GPU. d_packed from CustomObsCuda_PackArenas.
// d_obs: output, totalPlayers * CUSTOM_OBS_CUDA_OBS_SIZE floats
void CustomObsCuda_BuildBatch(
	const float* d_packed,
	const int* d_arenaPlayerStartIdx,
	int numArenas,
	int totalPlayers,
	float* d_obs,
	cudaStream_t stream = 0
);

// High-level: build obs for a batch of game states. Each state = one arena.
// Allocates d_packed and d_arenaPlayerStartIdx internally. d_obs must be pre-allocated.
// Returns totalPlayers (for validation).
int CustomObsCuda_BuildBatchFromStates(
	const std::vector<const void*>& gameStates,
	float* d_obs,
	cudaStream_t stream = 0
);

} // namespace GGL
#endif

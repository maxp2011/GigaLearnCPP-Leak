#pragma once

#ifdef RG_CUDA_SUPPORT

namespace GGL {
// Constants for CustomObs CUDA kernel - no host-only includes
constexpr int CUSTOM_OBS_CUDA_ARENA_STRIDE = 350;
constexpr int CUSTOM_OBS_CUDA_OBS_SIZE = 323;
constexpr int CUSTOM_OBS_CUDA_MAX_PLAYERS_PER_ARENA = 6;
constexpr int CUSTOM_OBS_CUDA_PLAYER_FEAT_SIZE = 44;
}

#endif

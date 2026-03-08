#include "CustomObsCuda.h"
#ifdef RG_CUDA_SUPPORT

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <RLGymCPP/CommonValues.h>
#include <cuda_runtime.h>
#include <cstring>

namespace GGL {

static bool HasFlipOrJump(float isOnGround, float hasFlipped, float hasDoubleJumped, float airTimeSinceJump) {
	constexpr float DOUBLEJUMP_MAX_DELAY = 1.25f;
	return isOnGround != 0.f || (hasFlipped == 0.f && hasDoubleJumped == 0.f && airTimeSinceJump < DOUBLEJUMP_MAX_DELAY);
}

void CustomObsCuda_PackArenas(
	const std::vector<const void*>& gameStates,
	const std::vector<int>& arenaPlayerStartIdx,
	float* d_packed,
	cudaStream_t stream
) {
	constexpr float INV_X = -1.f, INV_Y = -1.f, INV_Z = 1.f;
	int numArenas = (int)gameStates.size();
	std::vector<float> h_packed(numArenas * CUSTOM_OBS_CUDA_ARENA_STRIDE, 0.f);

	for (int a = 0; a < numArenas; a++) {
		const auto* gs = static_cast<const RLGC::GameState*>(gameStates[a]);
		float* arenaOut = h_packed.data() + a * CUSTOM_OBS_CUDA_ARENA_STRIDE;

		// Ball (0-8)
		arenaOut[0] = gs->ball.pos.x;
		arenaOut[1] = gs->ball.pos.y;
		arenaOut[2] = gs->ball.pos.z;
		arenaOut[3] = gs->ball.vel.x;
		arenaOut[4] = gs->ball.vel.y;
		arenaOut[5] = gs->ball.vel.z;
		arenaOut[6] = gs->ball.angVel.x;
		arenaOut[7] = gs->ball.angVel.y;
		arenaOut[8] = gs->ball.angVel.z;

		// Boost pads (9-42 blue, 43-76 orange) and timers (77-110 blue, 111-144 orange)
		const auto& pads = gs->GetBoostPads(false);
		const auto& timers = gs->GetBoostPadTimers(false);
		const auto& padsInv = gs->GetBoostPads(true);
		const auto& timersInv = gs->GetBoostPadTimers(true);
		for (int i = 0; i < 34; i++) {
			arenaOut[9 + i] = pads[i] ? 1.f : 0.f;
			arenaOut[43 + i] = padsInv[i] ? 1.f : 0.f;
			arenaOut[77 + i] = timers[i];
			arenaOut[111 + i] = timersInv[i];
		}

		// Num players (145)
		int np = (int)gs->players.size();
		arenaOut[145] = (float)np;

		// Players (146 + i*34), pad to 6
		for (int i = 0; i < 6; i++) {
			float* pOut = arenaOut + 146 + i * 34;
			if (i >= np) {
				std::memset(pOut, 0, 34 * sizeof(float));
				pOut[18] = 1.f; // isDemoed = true for padding
				continue;
			}
			const auto& pl = gs->players[i];
			// Store RAW (no inversion) - kernel inverts per viewer
			pOut[0] = pl.pos.x;
			pOut[1] = pl.pos.y;
			pOut[2] = pl.pos.z;
			pOut[3] = pl.vel.x;
			pOut[4] = pl.vel.y;
			pOut[5] = pl.vel.z;
			pOut[6] = pl.angVel.x;
			pOut[7] = pl.angVel.y;
			pOut[8] = pl.angVel.z;
			pOut[9] = pl.rotMat.forward.x;
			pOut[10] = pl.rotMat.forward.y;
			pOut[11] = pl.rotMat.forward.z;
			pOut[12] = pl.rotMat.up.x;
			pOut[13] = pl.rotMat.up.y;
			pOut[14] = pl.rotMat.up.z;
			pOut[15] = pl.boost;
			pOut[16] = pl.isOnGround ? 1.f : 0.f;
			pOut[17] = HasFlipOrJump(pl.isOnGround ? 1.f : 0.f, pl.hasFlipped ? 1.f : 0.f, pl.hasDoubleJumped ? 1.f : 0.f, pl.airTimeSinceJump) ? 1.f : 0.f;
			pOut[18] = pl.isDemoed ? 1.f : 0.f;
			pOut[19] = pl.hasJumped ? 1.f : 0.f;
			for (int j = 0; j < 8; j++)
				pOut[20 + j] = pl.prevAction[j];
			pOut[28] = (float)pl.carId;
			pOut[29] = (pl.team == RocketSim::Team::ORANGE) ? 1.f : 0.f;
			pOut[30] = pl.jumpTime;
			pOut[31] = pl.flipTime;
			pOut[32] = pl.airTimeSinceJump;
			pOut[33] = pl.timeSpentBoosting;
			pOut[34] = pl.handbrakeVal;
		}
	}

	cudaMemcpyAsync(d_packed, h_packed.data(), numArenas * CUSTOM_OBS_CUDA_ARENA_STRIDE * sizeof(float), cudaMemcpyHostToDevice, stream);
}

int CustomObsCuda_BuildBatchFromStates(
	const std::vector<const void*>& gameStates,
	float* d_obs,
	cudaStream_t stream
) {
	int numArenas = (int)gameStates.size();
	if (numArenas == 0) return 0;

	std::vector<int> arenaPlayerStartIdx;
	arenaPlayerStartIdx.reserve(numArenas + 1);
	int total = 0;
	arenaPlayerStartIdx.push_back(0);
	for (int a = 0; a < numArenas; a++) {
		const auto* gs = static_cast<const RLGC::GameState*>(gameStates[a]);
		total += (int)gs->players.size();
		arenaPlayerStartIdx.push_back(total);
	}

	float* d_packed = nullptr;
	int* d_arenaIdx = nullptr;
	cudaMalloc(&d_packed, numArenas * CUSTOM_OBS_CUDA_ARENA_STRIDE * sizeof(float));
	cudaMalloc(&d_arenaIdx, (numArenas + 1) * sizeof(int));
	cudaMemcpyAsync(d_arenaIdx, arenaPlayerStartIdx.data(), (numArenas + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

	CustomObsCuda_PackArenas(gameStates, arenaPlayerStartIdx, d_packed, stream);
	CustomObsCuda_BuildBatch(d_packed, d_arenaIdx, numArenas, total, d_obs, stream);

	cudaFree(d_packed);
	cudaFree(d_arenaIdx);
	return total;
}

} // namespace GGL
#endif

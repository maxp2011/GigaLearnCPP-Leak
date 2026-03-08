#pragma hd_warning_disable
#include "CustomObsCuda_constants.h"
#include <cuda_runtime.h>
#include <cmath>

#ifdef RG_CUDA_SUPPORT

namespace GGL {

// Constants matching CustomObs.h
__constant__ float kPosCoef;
__constant__ float kVelCoef;
__constant__ float kAngVelCoef;
__constant__ float kDistCoef;
__constant__ float kSpeedCoef;
__constant__ float kBallSpeedCoef;
__constant__ float kTimeCoef;
__constant__ float kBallMaxSpeed;
__constant__ float kGoalY;
__constant__ float kJumpTimeMax;
__constant__ float kFlipTimeMax;
__constant__ float kAirTimeMax;
__constant__ float kBoostTimeMax;

#define INV_VEC_X (-1.f)
#define INV_VEC_Y (-1.f)
#define INV_VEC_Z (1.f)

__device__ float dot3(float ax, float ay, float az, float bx, float by, float bz) {
	return ax * bx + ay * by + az * bz;
}

__device__ void cross3(float ax, float ay, float az, float bx, float by, float bz, float& rx, float& ry, float& rz) {
	rx = ay * bz - az * by;
	ry = az * bx - ax * bz;
	rz = ax * by - ay * bx;
}

__device__ float length3(float x, float y, float z) {
	return sqrtf(x * x + y * y + z * z);
}

__device__ float ball_goal_progress(float vy, bool inv) {
	float dir = inv ? -1.f : 1.f;
	return (vy * dir) * kBallSpeedCoef;
}

__device__ float time_to_ball(float dx, float dy, float dz, float vx, float vy, float vz) {
	float dist = length3(dx, dy, dz);
	float spd = length3(vx, vy, vz);
	return dist / (spd + 500.f);
}

__global__ void custom_obs_kernel(
	const float* __restrict__ packed,
	const int* __restrict__ arenaPlayerStartIdx,
	int numArenas,
	int totalPlayers,
	float* __restrict__ obs
) {
	int globalPlayerIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalPlayerIdx >= totalPlayers) return;

	// Find arena and local player index
	int arenaIdx = 0;
	while (arenaIdx < numArenas && arenaPlayerStartIdx[arenaIdx + 1] <= globalPlayerIdx)
		arenaIdx++;
	int localPlayerIdx = globalPlayerIdx - arenaPlayerStartIdx[arenaIdx];

	const float* arenaData = packed + arenaIdx * CUSTOM_OBS_CUDA_ARENA_STRIDE;

	// Ball (raw, we invert per-player based on team)
	float ballPosX = arenaData[0], ballPosY = arenaData[1], ballPosZ = arenaData[2];
	float ballVelX = arenaData[3], ballVelY = arenaData[4], ballVelZ = arenaData[5];
	float ballAngX = arenaData[6], ballAngY = arenaData[7], ballAngZ = arenaData[8];

	int numPlayers = (int)arenaData[145];
	if (localPlayerIdx >= numPlayers) {
		// Pad with zeros
		for (int i = 0; i < CUSTOM_OBS_CUDA_OBS_SIZE; i++)
			obs[globalPlayerIdx * CUSTOM_OBS_CUDA_OBS_SIZE + i] = 0.f;
		return;
	}

	// Self player data (offset 146 + localPlayerIdx * 34)
	const float* selfData = arenaData + 146 + localPlayerIdx * 34;
	float selfTeam = selfData[29]; // team: 0=blue, 1=orange
	bool inv = (selfTeam > 0.5f);

	// Invert ball for this player
	float ibx = ballPosX, iby = ballPosY, ibz = ballPosZ;
	float ibvx = ballVelX, ibvy = ballVelY, ibvz = ballVelZ;
	float ibax = ballAngX, ibay = ballAngY, ibaz = ballAngZ;
	if (inv) {
		ibx *= INV_VEC_X; iby *= INV_VEC_Y; ibz *= INV_VEC_Z;
		ibvx *= INV_VEC_X; ibvy *= INV_VEC_Y; ibvz *= INV_VEC_Z;
		ibax *= INV_VEC_X; ibay *= INV_VEC_Y; ibaz *= INV_VEC_Z;
	}

	// Goals
	float ownGoalX = 0.f, ownGoalY = inv ? kGoalY : -kGoalY, ownGoalZ = 321.f;
	float enemyGoalX = 0.f, enemyGoalY = inv ? -kGoalY : kGoalY, enemyGoalZ = 321.f;

	// Closest to ball: scan all players
	int closestGlobalId = -1;
	int closestBlueId = -1;
	int closestOrangeId = -1;
	float closestGlobalDist = 1e9f;
	float closestBlueDist = 1e9f;
	float closestOrangeDist = 1e9f;

	for (int i = 0; i < numPlayers; i++) {
		const float* p = arenaData + 146 + i * 34;
		if (p[18] > 0.5f) continue; // isDemoed
		float px = p[0], py = p[1], pz = p[2];
		float dx = ballPosX - px, dy = ballPosY - py, dz = ballPosZ - pz;
		float d = length3(dx, dy, dz);
		int carId = (int)p[30];
		float team = p[32];
		if (d < closestGlobalDist) {
			closestGlobalDist = d;
			closestGlobalId = carId;
		}
		if (team < 0.5f && d < closestBlueDist) {
			closestBlueDist = d;
			closestBlueId = carId;
		}
		if (team > 0.5f && d < closestOrangeDist) {
			closestOrangeDist = d;
			closestOrangeId = carId;
		}
	}

	int selfCarId = (int)selfData[28];
	bool selfClosestGlobal = (selfCarId == closestGlobalId);
	bool selfClosestTeam = (inv ? (selfCarId == closestOrangeId) : (selfCarId == closestBlueId));

	// Output pointer
	float* out = obs + globalPlayerIdx * CUSTOM_OBS_CUDA_OBS_SIZE;
	int idx = 0;

	// Ball section (17 floats)
	out[idx++] = ibx * kPosCoef;
	out[idx++] = iby * kPosCoef;
	out[idx++] = ibz * kPosCoef;
	out[idx++] = ibvx / kBallMaxSpeed;
	out[idx++] = ibvy / kBallMaxSpeed;
	out[idx++] = ibvz / kBallMaxSpeed;
	out[idx++] = ibax * kAngVelCoef;
	out[idx++] = ibay * kAngVelCoef;
	out[idx++] = ibaz * kAngVelCoef;
	out[idx++] = length3(ibvx, ibvy, ibvz) * kBallSpeedCoef;
	float ballToOwnX = ownGoalX - ibx, ballToOwnY = ownGoalY - iby, ballToOwnZ = ownGoalZ - ibz;
	float ballToEnemyX = enemyGoalX - ibx, ballToEnemyY = enemyGoalY - iby, ballToEnemyZ = enemyGoalZ - ibz;
	out[idx++] = ballToOwnX * kDistCoef;
	out[idx++] = ballToOwnY * kDistCoef;
	out[idx++] = ballToOwnZ * kDistCoef;
	out[idx++] = ballToEnemyX * kDistCoef;
	out[idx++] = ballToEnemyY * kDistCoef;
	out[idx++] = ballToEnemyZ * kDistCoef;
	out[idx++] = ball_goal_progress(ibvy, inv);

	// Prev action (8)
	for (int i = 0; i < 8; i++)
		out[idx++] = selfData[22 + i];

	// Boost pads (34) - use inv view
	int padBase = inv ? 43 : 9;
	int timerBase = inv ? 111 : 77;
	for (int i = 0; i < 34; i++) {
		if (arenaData[padBase + i] > 0.5f)
			out[idx++] = 1.f;
		else
			out[idx++] = expf(-arenaData[timerBase + i] * 0.2f);
	}

	// Self (44)
	float spx = selfData[0], spy = selfData[1], spz = selfData[2];
	float sfx = selfData[9], sfy = selfData[10], sfz = selfData[11];
	float sux = selfData[12], suy = selfData[13], suz = selfData[14];
	float svx = selfData[3], svy = selfData[4], svz = selfData[5];
	float sax = selfData[6], say = selfData[7], saz = selfData[8];
	if (inv) {
		spx *= INV_VEC_X; spy *= INV_VEC_Y; spz *= INV_VEC_Z;
		sfx *= INV_VEC_X; sfy *= INV_VEC_Y; sfz *= INV_VEC_Z;
		sux *= INV_VEC_X; suy *= INV_VEC_Y; suz *= INV_VEC_Z;
		svx *= INV_VEC_X; svy *= INV_VEC_Y; svz *= INV_VEC_Z;
		sax *= INV_VEC_X; say *= INV_VEC_Y; saz *= INV_VEC_Z;
	}
	float jumpTime = selfData[30];
	float flipTime = selfData[31];
	float airTimeSinceJump = selfData[32];
	float timeSpentBoosting = selfData[33];
	float handbrakeVal = selfData[34];

	// 29 base
	out[idx++] = spx * kPosCoef;
	out[idx++] = spy * kPosCoef;
	out[idx++] = spz * kPosCoef;
	out[idx++] = sfx; out[idx++] = sfy; out[idx++] = sfz;
	out[idx++] = sux; out[idx++] = suy; out[idx++] = suz;
	out[idx++] = svx * kVelCoef; out[idx++] = svy * kVelCoef; out[idx++] = svz * kVelCoef;
	out[idx++] = sax * kAngVelCoef; out[idx++] = say * kAngVelCoef; out[idx++] = saz * kAngVelCoef;
	// Local ang vel: forward.dot(angVel), right.dot(angVel), up.dot(angVel); right = forward x up
	float srx, sry, srz;
	cross3(sfx, sfy, sfz, sux, suy, suz, srx, sry, srz);
	out[idx++] = (sfx * sax + sfy * say + sfz * saz) * kAngVelCoef;
	out[idx++] = (srx * sax + sry * say + srz * saz) * kAngVelCoef;
	out[idx++] = (sux * sax + suy * say + suz * saz) * kAngVelCoef;
	float srelx = ibx - spx, srely = iby - spy, srelz = ibz - spz;
	float srvx = ibvx - svx, srvy = ibvy - svy, srvz = ibvz - svz;
	out[idx++] = (sfx * srelx + sfy * srely + sfz * srelz) * kPosCoef;
	out[idx++] = (sfx * srvx + sfy * srvy + sfz * srvz) * kVelCoef;
	// Player layout: 0-2 pos, 3-5 vel, 6-8 angVel, 9-11 forward, 12-14 up, 15 boost, 16 isOnGround, 17 hasFlipOrJump, 18 isDemoed, 19 hasJumped, 20-27 prevAction, 28 carId, 29 team, 30-34 dash
	out[idx++] = selfData[15] / 100.f;
	out[idx++] = selfData[16];
	out[idx++] = selfData[17];
	out[idx++] = selfData[18];
	out[idx++] = selfData[19];
	// 9 custom
	out[idx++] = length3(ibx - spx, iby - spy, ibz - spz) * kDistCoef;
	out[idx++] = length3(svx, svy, svz) * kSpeedCoef;
	out[idx++] = length3(ownGoalX - spx, ownGoalY - spy, ownGoalZ - spz) * kDistCoef;
	out[idx++] = length3(enemyGoalX - spx, enemyGoalY - spy, enemyGoalZ - spz) * kDistCoef;
	out[idx++] = fminf(time_to_ball(ibx - spx, iby - spy, ibz - spz, svx, svy, svz) * kTimeCoef, 1.f);
	out[idx++] = selfClosestGlobal ? 1.f : 0.f;
	out[idx++] = selfClosestTeam ? 1.f : 0.f;
	float toBallLen = length3(srelx, srely, srelz);
	float align = 0.f;
	if (toBallLen > 1.f) {
		float il = 1.f / toBallLen;
		align = dot3(sfx, sfy, sfz, srelx * il, srely * il, srelz * il);
	}
	out[idx++] = align;
	out[idx++] = ball_goal_progress(ibvy, inv);
	// 5 dash
	out[idx++] = fminf(jumpTime / kJumpTimeMax, 1.f);
	out[idx++] = fminf(flipTime / kFlipTimeMax, 1.f);
	out[idx++] = fminf(airTimeSinceJump / kAirTimeMax, 1.f);
	out[idx++] = fminf(timeSpentBoosting / kBoostTimeMax, 1.f);
	out[idx++] = fminf(fmaxf(handbrakeVal, 0.f), 1.f);
	out[idx++] = 0.f; // team flag self

	// Teammates (2 * 44 = 88) and Opponents (3 * 44 = 132)
	float teammates[88];
	float opponents[132];
	int nTM = 0, nOpp = 0;
	for (int i = 0; i < numPlayers; i++) {
		if (i == localPlayerIdx) continue;
		const float* p = arenaData + 146 + i * 34;
		if (p[18] > 0.5f) continue; // isDemoed
		bool isTM = (p[32] == selfTeam);
		bool otherClosestGlobal = ((int)p[28] == closestGlobalId);
		bool otherClosestTeam = isTM ? ((int)p[28] == (inv ? closestOrangeId : closestBlueId)) : ((int)p[28] == (inv ? closestBlueId : closestOrangeId));
		float teamFlag = isTM ? 1.f : -1.f;

		float tpx = p[0], tpy = p[1], tpz = p[2];
		float tfx = p[9], tfy = p[10], tfz = p[11];
		float tux = p[12], tuy = p[13], tuz = p[14];
		float tvx = p[3], tvy = p[4], tvz = p[5];
		float tax = p[6], tay = p[7], taz = p[8];
		if (inv) {
			tpx *= INV_VEC_X; tpy *= INV_VEC_Y; tpz *= INV_VEC_Z;
			tfx *= INV_VEC_X; tfy *= INV_VEC_Y; tfz *= INV_VEC_Z;
			tux *= INV_VEC_X; tuy *= INV_VEC_Y; tuz *= INV_VEC_Z;
			tvx *= INV_VEC_X; tvy *= INV_VEC_Y; tvz *= INV_VEC_Z;
			tax *= INV_VEC_X; tay *= INV_VEC_Y; taz *= INV_VEC_Z;
		}
		float tBoost = p[15];
		float tJump = p[30], tFlip = p[31], tAir = p[32], tBoostTime = p[33], tHand = p[34];
		float hasFlip = p[17];

		float* dest = isTM ? (teammates + nTM * 44) : (opponents + nOpp * 44);
		int didx = 0;
		// 29 base
		dest[didx++] = tpx * kPosCoef;
		dest[didx++] = tpy * kPosCoef;
		dest[didx++] = tpz * kPosCoef;
		dest[didx++] = tfx; dest[didx++] = tfy; dest[didx++] = tfz;
		dest[didx++] = tux; dest[didx++] = tuy; dest[didx++] = tuz;
		dest[didx++] = tvx * kVelCoef; dest[didx++] = tvy * kVelCoef; dest[didx++] = tvz * kVelCoef;
		dest[didx++] = tax * kAngVelCoef; dest[didx++] = tay * kAngVelCoef; dest[didx++] = taz * kAngVelCoef;
		float trx, tryy, trz;
		cross3(tfx, tfy, tfz, tux, tuy, tuz, trx, tryy, trz);
		dest[didx++] = (tfx * tax + tfy * tay + tfz * taz) * kAngVelCoef;
		dest[didx++] = (trx * tax + tryy * tay + trz * taz) * kAngVelCoef;
		dest[didx++] = (tux * tax + tuy * tay + tuz * taz) * kAngVelCoef;
		float trpx = ibx - tpx, trpy = iby - tpy, trpz = ibz - tpz;
		float trvx = ibvx - tvx, trvy = ibvy - tvy, trvz = ibvz - tvz;
		dest[didx++] = (tfx * trpx + tfy * trpy + tfz * trpz) * kPosCoef;
		dest[didx++] = (tfx * trvx + tfy * trvy + tfz * trvz) * kVelCoef;
		dest[didx++] = tBoost / 100.f;
		dest[didx++] = p[16];
		dest[didx++] = hasFlip;
		dest[didx++] = p[18];
		dest[didx++] = p[19];
		// 9 custom + 5 dash + 1 team
		dest[didx++] = length3(ibx - tpx, iby - tpy, ibz - tpz) * kDistCoef;
		dest[didx++] = length3(tvx, tvy, tvz) * kSpeedCoef;
		dest[didx++] = length3(ownGoalX - tpx, ownGoalY - tpy, ownGoalZ - tpz) * kDistCoef;
		dest[didx++] = length3(enemyGoalX - tpx, enemyGoalY - tpy, enemyGoalZ - tpz) * kDistCoef;
		dest[didx++] = fminf(time_to_ball(ibx - tpx, iby - tpy, ibz - tpz, tvx, tvy, tvz) * kTimeCoef, 1.f);
		dest[didx++] = otherClosestGlobal ? 1.f : 0.f;
		dest[didx++] = otherClosestTeam ? 1.f : 0.f;
		float tol = length3(trpx, trpy, trpz);
		float talign = 0.f;
		if (tol > 1.f) talign = dot3(tfx, tfy, tfz, trpx / tol, trpy / tol, trpz / tol);
		dest[didx++] = talign;
		dest[didx++] = ball_goal_progress(ibvy, inv);
		dest[didx++] = fminf(tJump / kJumpTimeMax, 1.f);
		dest[didx++] = fminf(tFlip / kFlipTimeMax, 1.f);
		dest[didx++] = fminf(tAir / kAirTimeMax, 1.f);
		dest[didx++] = fminf(tBoostTime / kBoostTimeMax, 1.f);
		dest[didx++] = fminf(fmaxf(tHand, 0.f), 1.f);
		dest[didx++] = teamFlag;

		if (isTM && nTM < 2) nTM++;
		if (!isTM && nOpp < 3) nOpp++;
	}

	// Copy teammates (pad to 2)
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 44; j++)
			out[idx++] = (i < nTM) ? teammates[i * 44 + j] : 0.f;
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 44; j++)
			out[idx++] = (i < nOpp) ? opponents[i * 44 + j] : 0.f;
	}
}

void CustomObsCuda_BuildBatch(
	const float* d_packed,
	const int* d_arenaPlayerStartIdx,
	int numArenas,
	int totalPlayers,
	float* d_obs,
	cudaStream_t stream
) {
	// Set constants
	float hPosCoef = 1.f / 5000.f;
	float hVelCoef = 1.f / 2300.f;
	float hAngVelCoef = 1.f / 3.f;
	float hDistCoef = 1.f / 3000.f;
	float hSpeedCoef = 1.f / 2300.f;
	float hBallSpeedCoef = 1.f / 6000.f;
	float hTimeCoef = 1.f / 5.f;
	float hBallMaxSpeed = 6000.f;
	float hGoalY = 5120.f;
	float hJumpTimeMax = 0.2f;
	float hFlipTimeMax = 0.65f;
	float hAirTimeMax = 1.25f;
	float hBoostTimeMax = 0.1f;

	cudaMemcpyToSymbol(kPosCoef, &hPosCoef, sizeof(float));
	cudaMemcpyToSymbol(kVelCoef, &hVelCoef, sizeof(float));
	cudaMemcpyToSymbol(kAngVelCoef, &hAngVelCoef, sizeof(float));
	cudaMemcpyToSymbol(kDistCoef, &hDistCoef, sizeof(float));
	cudaMemcpyToSymbol(kSpeedCoef, &hSpeedCoef, sizeof(float));
	cudaMemcpyToSymbol(kBallSpeedCoef, &hBallSpeedCoef, sizeof(float));
	cudaMemcpyToSymbol(kTimeCoef, &hTimeCoef, sizeof(float));
	cudaMemcpyToSymbol(kBallMaxSpeed, &hBallMaxSpeed, sizeof(float));
	cudaMemcpyToSymbol(kGoalY, &hGoalY, sizeof(float));
	cudaMemcpyToSymbol(kJumpTimeMax, &hJumpTimeMax, sizeof(float));
	cudaMemcpyToSymbol(kFlipTimeMax, &hFlipTimeMax, sizeof(float));
	cudaMemcpyToSymbol(kAirTimeMax, &hAirTimeMax, sizeof(float));
	cudaMemcpyToSymbol(kBoostTimeMax, &hBoostTimeMax, sizeof(float));

	int blockSize = 256;
	int numBlocks = (totalPlayers + blockSize - 1) / blockSize;
	custom_obs_kernel<<<numBlocks, blockSize, 0, stream>>>(
		d_packed, d_arenaPlayerStartIdx, numArenas, totalPlayers, d_obs
	);
}

} // namespace GGL
#endif

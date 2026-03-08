#pragma hd_warning_disable
#include <cuda_runtime.h>

// TerminalType: NOT_TERMINAL=0, NORMAL=1, TRUNCATED=2
#define TERM_NORMAL 1
#define TERM_TRUNCATED 2

__global__ void gae_kernel(
	const float* __restrict__ rews,
	const int8_t* __restrict__ terminals,
	const float* __restrict__ valPreds,
	const float* __restrict__ truncValPreds,
	int numReturns,
	int numTruncs,
	float gamma,
	float lambda,
	float returnStd,
	float clipRange,
	float* __restrict__ outReturns,
	float* __restrict__ outAdvantages,
	float* __restrict__ totalRew,
	float* __restrict__ totalClippedRew
) {
	// Single-thread backward scan (inherently sequential)
	if (threadIdx.x != 0 || blockIdx.x != 0) return;

	float prevLambda = 0.f;
	float prevRet = 0.f;
	int truncCount = 0;
	float sumRew = 0.f;
	float sumClippedRew = 0.f;

	for (int step = numReturns - 1; step >= 0; step--) {
		int8_t terminal = terminals[step];
		float done = (terminal == TERM_NORMAL) ? 1.f : 0.f;
		float trunc = (terminal == TERM_TRUNCATED) ? 1.f : 0.f;

		float curReward = rews[step];
		if (returnStd != 0.f) {
			curReward /= returnStd;
			sumRew += fabsf(curReward);
			if (clipRange > 0.f) {
				curReward = fmaxf(-clipRange, fminf(clipRange, curReward));
			}
			sumClippedRew += fabsf(curReward);
		} else {
			sumRew += fabsf(curReward);
		}

		float nextValPred;
		if (terminal == TERM_TRUNCATED) {
			nextValPred = truncValPreds[truncCount];
			truncCount++;
		} else {
			nextValPred = (step + 1 < numReturns) ? valPreds[step + 1] : 0.f;
		}

		float predReturn = curReward + gamma * nextValPred * (1.f - done);
		float delta = predReturn - valPreds[step];
		float curReturn = rews[step] + prevRet * gamma * (1.f - done) * (1.f - trunc);

		outReturns[step] = curReturn;
		prevLambda = delta + gamma * lambda * (1.f - done) * (1.f - trunc) * prevLambda;
		outAdvantages[step] = prevLambda;
		prevRet = curReturn;
	}

	*totalRew = sumRew;
	*totalClippedRew = sumClippedRew;
}

extern "C" void gae_cuda_launch(
	const float* rews,
	const int8_t* terminals,
	const float* valPreds,
	const float* truncValPreds,
	int numReturns,
	int numTruncs,
	float gamma,
	float lambda,
	float returnStd,
	float clipRange,
	float* outReturns,
	float* outAdvantages,
	float* totalRew,
	float* totalClippedRew,
	cudaStream_t stream
) {
	gae_kernel<<<1, 1, 0, stream>>>(
		rews, terminals, valPreds,
		truncValPreds ? truncValPreds : (const float*)nullptr,
		numReturns, numTruncs,
		gamma, lambda, returnStd, clipRange,
		outReturns, outAdvantages, totalRew, totalClippedRew
	);
}

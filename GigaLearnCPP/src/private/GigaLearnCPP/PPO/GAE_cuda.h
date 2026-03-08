#pragma once

#ifdef RG_CUDA_SUPPORT
#include <cuda_runtime.h>

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
);
#endif

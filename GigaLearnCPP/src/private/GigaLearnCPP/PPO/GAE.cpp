#include "GAE.h"
#ifdef RG_CUDA_SUPPORT
#include "GAE_cuda.h"
#include <c10/cuda/CUDAStream.h>
#endif

void GGL::GAE::Compute(
	torch::Tensor rews, torch::Tensor terminals, torch::Tensor valPreds, torch::Tensor truncValPreds,
	torch::Tensor& outAdvantages, torch::Tensor& outTargetValues, torch::Tensor& outReturns, float& outRewClipPortion,
	float gamma, float lambda, float returnStd, float clipRange
) {
	bool hasTruncValPreds = truncValPreds.defined();
	int numReturns = (int)rews.size(0);

	rews = rews.contiguous();
	terminals = terminals.contiguous();
	valPreds = valPreds.contiguous();
	if (hasTruncValPreds)
		truncValPreds = truncValPreds.contiguous();

#ifdef RG_CUDA_SUPPORT
	if (rews.is_cuda()) {
		int numTruncs = hasTruncValPreds ? (int)truncValPreds.size(0) : 0;
		if (hasTruncValPreds && truncValPreds.numel() == 0)
			numTruncs = 0;

		outReturns = torch::empty({ numReturns }, rews.options());
		outAdvantages = torch::empty({ numReturns }, rews.options());
		auto dTotalRew = torch::empty({ 1 }, rews.options());
		auto dTotalClippedRew = torch::empty({ 1 }, rews.options());

		cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
		gae_cuda_launch(
			rews.const_data_ptr<float>(),
			terminals.const_data_ptr<int8_t>(),
			valPreds.const_data_ptr<float>(),
			hasTruncValPreds ? truncValPreds.const_data_ptr<float>() : nullptr,
			numReturns, numTruncs,
			gamma, lambda, returnStd, clipRange,
			outReturns.data_ptr<float>(),
			outAdvantages.data_ptr<float>(),
			dTotalRew.data_ptr<float>(),
			dTotalClippedRew.data_ptr<float>(),
			stream
		);
		cudaStreamSynchronize(stream);

		float totalRew = dTotalRew.cpu().item<float>();
		float totalClippedRew = dTotalClippedRew.cpu().item<float>();
		outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
		outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
		return;
	}
#endif

	// CPU path
	float prevLambda = 0;
	float prevRet = 0;
	int truncCount = 0;
	float totalRew = 0, totalClippedRew = 0;

	auto _terminals = terminals.const_data_ptr<int8_t>();
	auto _rews = rews.const_data_ptr<float>();
	auto _valPreds = valPreds.const_data_ptr<float>();

	const float* _truncValPreds = nullptr;
	int numTruncs = 0;
	if (hasTruncValPreds) {
		_truncValPreds = truncValPreds.const_data_ptr<float>();
		numTruncs = (int)truncValPreds.size(0);
	}

	std::vector<float> _outReturns(numReturns, 0);
	std::vector<float> _outAdvantages(numReturns, 0);

	for (int step = numReturns - 1; step >= 0; step--) {
		uint8_t terminal = _terminals[step];
		float done = (terminal == RLGC::TerminalType::NORMAL) ? 1.f : 0.f;
		float trunc = (terminal == RLGC::TerminalType::TRUNCATED) ? 1.f : 0.f;

		float curReward;
		if (returnStd != 0) {
			curReward = _rews[step] / returnStd;
			totalRew += (float)fabs(curReward);
			if (clipRange > 0)
				curReward = RS_CLAMP(curReward, -clipRange, clipRange);
			totalClippedRew += (float)fabs(curReward);
		} else {
			curReward = _rews[step];
			totalRew += (float)fabs(curReward);
		}

		float nextValPred;
		if (terminal == RLGC::TerminalType::TRUNCATED) {
			if (!hasTruncValPreds)
				RG_ERR_CLOSE("GAE encountered a truncated terminal, but has no truncated val pred");
			if (truncCount >= numTruncs)
				RG_ERR_CLOSE("GAE encountered too many truncated terminals, not enough val preds (max: " << numTruncs << ")");
			nextValPred = _truncValPreds[truncCount];
			truncCount++;
		} else {
			nextValPred = (step + 1 < numReturns) ? _valPreds[step + 1] : 0.f;
		}

		float predReturn = curReward + gamma * nextValPred * (1.f - done);
		float delta = predReturn - _valPreds[step];
		float curReturn = _rews[step] + prevRet * gamma * (1.f - done) * (1.f - trunc);
		_outReturns[step] = curReturn;
		prevLambda = delta + gamma * lambda * (1.f - done) * (1.f - trunc) * prevLambda;
		_outAdvantages[step] = prevLambda;
		prevRet = curReturn;
	}

	if (hasTruncValPreds && truncCount != (int)truncValPreds.size(0))
		RG_ERR_CLOSE("GAE didn't receive expected truncation count (only " << truncCount << "/" << truncValPreds.size(0) << ")");

	outReturns = torch::tensor(_outReturns, rews.options());
	outAdvantages = torch::tensor(_outAdvantages, rews.options());
	outTargetValues = valPreds.slice(0, 0, numReturns) + outAdvantages;
	outRewClipPortion = (totalRew - totalClippedRew) / RS_MAX(totalRew, 1e-7f);
}
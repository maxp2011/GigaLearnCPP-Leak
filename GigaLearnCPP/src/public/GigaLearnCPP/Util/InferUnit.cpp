#include "InferUnit.h"

#include <GigaLearnCPP/Util/Models.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#ifdef RG_CUDA_SUPPORT
#include <GigaLearnCPP/ObsBuilders/CustomObsCuda.h>
#include <c10/cuda/CUDAStream.h>
#endif

GGL::InferUnit::InferUnit(
	RLGC::ObsBuilder* obsBuilder, int obsSize, RLGC::ActionParser* actionParser,
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, 
	std::filesystem::path modelsFolder, bool useGPU) : 
	obsBuilder(obsBuilder), obsSize(obsSize), actionParser(actionParser), useGPU(useGPU) {

	this->models = new ModelSet();

	try {
		PPOLearner::MakeModels(
			false, obsSize, actionParser->GetActionAmount(),
			sharedHeadConfig, policyConfig, {},
			useGPU ? torch::kCUDA : torch::kCPU,
			*this->models
		);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to construct models: " << e.what());
	}

	try {
		this->models->Load(modelsFolder, false, false);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when trying to load models: " << e.what());
	}
}

RLGC::Action GGL::InferUnit::InferAction(const RLGC::Player& player, const RLGC::GameState& state, bool deterministic, float temperature) {
	return BatchInferActions({ player }, { state }, deterministic, temperature)[0];
}

std::vector<RLGC::Action> GGL::InferUnit::BatchInferActions(const std::vector<RLGC::Player>& players, const std::vector<RLGC::GameState>& states, bool deterministic, float temperature) {
	RG_ASSERT(players.size() > 0 && states.size() > 0);
	RG_ASSERT(players.size() == states.size());

	int batchSize = players.size();
	std::vector<uint8_t> allActionMasks;
	for (int i = 0; i < batchSize; i++) {
		allActionMasks += actionParser->GetActionMask(players[i], states[i]);
	}

	torch::Tensor tObs;
#ifdef RG_CUDA_SUPPORT
	bool usedCudaObs = false;
	// CUDA path when CustomObs (323): use unique states, output order matches "all players from all games"
		if (useGPU && obsSize == 323) { // CustomObs size
		std::vector<const void*> uniqueStates;
		for (size_t i = 0; i < states.size(); i++) {
			if (i == 0 || &states[i] != &states[i - 1])
				uniqueStates.push_back(&states[i]);
		}
		int totalExpected = 0;
		for (const auto* p : uniqueStates)
			totalExpected += static_cast<const RLGC::GameState*>(p)->players.size();
		if (totalExpected == batchSize) {
			tObs = torch::empty({(int64_t)batchSize, obsSize}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
			int total = GGL::CustomObsCuda_BuildBatchFromStates(uniqueStates, tObs.data_ptr<float>(), c10::cuda::getCurrentCUDAStream());
			if (total == batchSize) usedCudaObs = true;
		}
	}
	if (!usedCudaObs)
#endif
	{
		std::vector<float> allObs;
		for (int i = 0; i < batchSize; i++) {
			FList curObs = obsBuilder->BuildObs(players[i], states[i]);
			if (curObs.size() != obsSize) {
				RG_ERR_CLOSE(
					"InferUnit: Obs builder produced an obs that differs from the provided size (expected: " << obsSize << ", got: " << curObs.size() << ")\n" <<
					"Make sure you provided the correct obs size to the InferUnit constructor.\n" <<
					"Also, make sure there aren't an incorrect number of players (there are " << states[i].players.size() << " in this state)"
				);
			}
			allObs += curObs;
		}
		tObs = torch::tensor(allObs).reshape({(int64_t)batchSize, obsSize});
		if (useGPU) tObs = tObs.to(torch::kCUDA);
	}

	std::vector<RLGC::Action> results = {};

	try {
		RG_NO_GRAD;

		auto device = useGPU ? torch::kCUDA : torch::kCPU;

		auto tActionMasks = torch::tensor(allActionMasks).reshape({(int64_t)batchSize, this->actionParser->GetActionAmount()});
		tActionMasks = tActionMasks.to(device);
		torch::Tensor tActions, tLogProbs;

		PPOLearner::InferActionsFromModels(*models, tObs, tActionMasks, deterministic, temperature, false, &tActions, &tLogProbs);

		auto actionIndices = TENSOR_TO_VEC<int>(tActions);
		
		for (int i = 0; i < batchSize; i++) 
			results.push_back(actionParser->ParseAction(actionIndices[i], players[i], states[i]));

	} catch (std::exception& e) {
		RG_ERR_CLOSE("InferUnit: Exception when inferring model: " << e.what());
	}

	return results;
}
#include "PPOLearner.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <public/GigaLearnCPP/Util/AvgTracker.h>

using namespace torch;

GGL::PPOLearner::PPOLearner(int obsSize, int numActions, PPOLearnerConfig _config, Device _device) : config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize (" << config.batchSize << ") must be a multiple of config.miniBatchSize (" << config.miniBatchSize << ")");

	MakeModels(true, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, models);

	SetLearningRates(config.policyLR, config.criticLR);

	// Print param counts
	RG_LOG("Model parameter counts:");
	uint64_t total = 0;
	for (auto model : this->models) {
		uint64_t count = model->GetParamCount();
		RG_LOG("\t\"" << model->modelName << "\": " << Utils::NumToStr(count));
		total += count;
	}
	RG_LOG("\t[Total]: " << Utils::NumToStr(total));

	if (config.useGuidingPolicy) {
		RG_LOG("Guiding policy enabled, loading from " << config.guidingPolicyPath << "...");
		MakeModels(false, obsSize, numActions, config.sharedHead, config.policy, config.critic, device, guidingPolicyModels);
		guidingPolicyModels.Load(config.guidingPolicyPath, false, false);
	}
}

void GGL::PPOLearner::MakeModels(
	bool makeCritic,
	int obsSize, int numActions, 
	PartialModelConfig sharedHeadConfig, PartialModelConfig policyConfig, PartialModelConfig criticConfig,
	torch::Device device, 
	ModelSet& outModels) {

	ModelConfig fullPolicyConfig = policyConfig;
	fullPolicyConfig.numInputs = obsSize;
	fullPolicyConfig.numOutputs = numActions;

	ModelConfig fullCriticConfig = criticConfig;
	fullCriticConfig.numInputs = obsSize;
	fullCriticConfig.numOutputs = 1;

	if (sharedHeadConfig.IsValid()) {

		ModelConfig fullSharedHeadConfig = sharedHeadConfig;
		fullSharedHeadConfig.numInputs = obsSize;
		fullSharedHeadConfig.numOutputs = 0;

		RG_ASSERT(!sharedHeadConfig.addOutputLayer);

		fullPolicyConfig.numInputs = fullSharedHeadConfig.layerSizes.back();
		fullCriticConfig.numInputs = fullSharedHeadConfig.layerSizes.back();

		outModels.Add(new Model("shared_head", fullSharedHeadConfig, device));
	}

	outModels.Add(new Model("policy", fullPolicyConfig, device));

	if (makeCritic)
		outModels.Add(new Model("critic", fullCriticConfig, device));
}

torch::Tensor GGL::PPOLearner::InferPolicyProbsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks,
	float temperature, bool halfPrec) {

	actionMasks = actionMasks.to(torch::kBool);

	constexpr float ACTION_MIN_PROB = 1e-11f;
	constexpr float ACTION_DISABLED_LOGIT = -1e10f;

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, halfPrec);

	auto logits = models["policy"]->Forward(obs, halfPrec) / temperature;

	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);
}

void GGL::PPOLearner::InferActionsFromModels(
	ModelSet& models,
	torch::Tensor obs, torch::Tensor actionMasks, 
	bool deterministic, float temperature, bool halfPrec,
	torch::Tensor* outActions, torch::Tensor* outLogProbs) {

	auto probs = InferPolicyProbsFromModels(models, obs, actionMasks, temperature, halfPrec);

	if (deterministic) {
		auto action = probs.argmax(1);
		if (outActions)
			*outActions = action.flatten();
	} else {
		auto action = torch::multinomial(probs, 1, true);
		auto logProb = torch::log(probs).gather(-1, action);
		if (outActions)
			*outActions = action.flatten();

		if (outLogProbs)
			*outLogProbs = logProb.flatten();
	}
}

void GGL::PPOLearner::InferActions(torch::Tensor obs, torch::Tensor actionMasks, torch::Tensor* outActions, torch::Tensor* outLogProbs, ModelSet* models, bool forceDeterministic) {
	bool useDeterministic = forceDeterministic || config.deterministic;
	InferActionsFromModels(models ? *models : this->models, obs, actionMasks, useDeterministic, config.policyTemperature, config.useHalfPrecision, outActions, outLogProbs);
}

torch::Tensor GGL::PPOLearner::InferCritic(torch::Tensor obs) {

	if (models["shared_head"])
		obs = models["shared_head"]->Forward(obs, config.useHalfPrecision);

	return models["critic"]->Forward(obs, config.useHalfPrecision).flatten();
}

// rocket-learn 1:1: raw entropy -sum(p*log(p)), no normalization. ent_coef applied in loss.
torch::Tensor ComputeEntropy(torch::Tensor probs, torch::Tensor actionMasks, bool maskEntropy) {
	auto entropy = -(probs.log() * probs).sum(-1);  // Raw Categorical entropy
	if (maskEntropy) {
		constexpr float DIVISOR_MIN = 1e-6f;
		auto maskSum = actionMasks.to(torch::kFloat32).sum(-1).clamp(1.001f, 1e9f);
		entropy /= maskSum.log().clamp(DIVISOR_MIN, 1000.f);
	}
	return entropy.clamp(0.f, 100.f).mean();  // Clamp to avoid nan
}

void GGL::PPOLearner::Learn(ExperienceBuffer& experience, Report& report, bool isFirstIteration) {
	auto mseLoss = torch::nn::MSELoss();
	bool useGpuAccum = device.is_cuda();

	// GPU: accumulate on device to avoid per-minibatch syncs; CPU: use MutAvgTracker
	MutAvgTracker avgEntropy, avgDivergence, avgPolicyLoss, avgCriticLoss, avgGuidingLoss, avgRatio, avgClip;
	torch::Tensor tEntropySum, tPolicyLossSum, tCriticLossSum, tDivergenceSum, tClipSum, tRatioSum, tGuidingLossSum;
	int64_t tCount = 0;
	int64_t tEntropyCount = 0;  // total elements (for entropy mean over samples)
	if (useGpuAccum) {
		tEntropySum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tPolicyLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tCriticLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tDivergenceSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tClipSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tRatioSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
		tGuidingLossSum = torch::zeros(1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
	}

	// Save parameters first
	auto policyBefore = models["policy"]->CopyParams();
	auto criticBefore = models["critic"]->CopyParams();
	torch::Tensor sharedHeadBefore;
	if (models["shared_head"])
		sharedHeadBefore = models["shared_head"]->CopyParams();

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;
	bool trainSharedHead = models["shared_head"] && (trainPolicy || trainCritic);

	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = experience.GetAllBatchesShuffled(config.batchSize, config.overbatching);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchActionMasks = batch.actionMasks;
			auto batchTargetValues = batch.targetValues;
			auto batchAdvantages = batch.advantages;

			auto fnRunMinibatch = [&](int start, int stop) {

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				auto actionMasks = batchActionMasks.slice(0, start, stop).to(device, true, true);
				
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

				// rocket-learn: adv = returns - V, then normalize per minibatch
				auto vals = InferCritic(obs);
				vals = vals.view_as(targetValues);
				auto advantages = (targetValues - vals.detach() - (targetValues - vals.detach()).mean())
					/ ((targetValues - vals.detach()).std() + 1e-8f);

				torch::Tensor probs, logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {

					// Get policy log probs and entropy
					{
						probs = InferPolicyProbsFromModels(models, obs, actionMasks, config.policyTemperature, false);
						logProbs = probs.log().gather(-1, acts.unsqueeze(-1));
						entropy = ComputeEntropy(probs, actionMasks, config.maskEntropy);
						if (useGpuAccum) {
							tEntropySum += entropy.detach().sum();
							tEntropyCount += entropy.numel();
						} else {
							avgEntropy += entropy.detach().cpu().item<float>();
						}
					}

					logProbs = logProbs.view_as(oldProbs);

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					if (useGpuAccum) {
						tRatioSum += ratio.detach().mean();
					} else {
						avgRatio += ratio.mean().detach().cpu().item<float>();
					}
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					if (useGpuAccum) {
						tPolicyLossSum += policyLoss.detach();
					} else {
						avgPolicyLoss += policyLoss.detach().cpu().item<float>();
					}

					ppoLoss = (policyLoss - entropy * config.entropyScale) * batchSizeRatio;

					if (config.useGuidingPolicy) {
						torch::Tensor guidingProbs;
						{
							RG_NO_GRAD;
							guidingProbs = InferPolicyProbsFromModels(guidingPolicyModels, obs, actionMasks, config.policyTemperature, config.useHalfPrecision);
						}

						auto guidingLoss = (guidingProbs - probs).abs().mean();
						if (useGpuAccum) {
							tGuidingLossSum += guidingLoss.detach();
						} else {
							avgGuidingLoss.Add(guidingLoss.detach().cpu().item<float>());
						}
						guidingLoss = guidingLoss * config.guidingStrength;
						ppoLoss = ppoLoss + guidingLoss;
					}
				}

				torch::Tensor criticLoss;
				if (trainCritic) {
					// vals already computed above for advantage normalization
					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					if (useGpuAccum) {
						tCriticLossSum += criticLoss.detach();
					} else {
						avgCriticLoss += criticLoss.detach().cpu().item<float>();
					}
				}

				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting;
					{
						RG_NO_GRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						if (useGpuAccum) {
							tDivergenceSum += klTensor.mean().detach();
						} else {
							avgDivergence += klTensor.mean().detach().cpu().item<float>();
						}

						auto clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat));
						if (useGpuAccum) {
							tClipSum += clipFraction.detach();
						} else {
							avgClip += clipFraction.cpu().item<float>();
						}
					}
				}
				if (useGpuAccum) tCount++;

				if (trainPolicy && trainCritic) {
					auto combinedLoss = ppoLoss + criticLoss;
					combinedLoss.backward();
				} else {
					if (trainPolicy)
						ppoLoss.backward();
					if (trainCritic)
						criticLoss.backward();
				}

				// rocket-learn: clip + step after each minibatch (not accumulated)
				if (trainPolicy)
					nn::utils::clip_grad_norm_(models["policy"]->parameters(), 0.5f);
				if (trainCritic)
					nn::utils::clip_grad_norm_(models["critic"]->parameters(), 0.5f);
				if (trainSharedHead)
					nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);
				models.StepOptims();
			};

			if (device.is_cpu()) {
				fnRunMinibatch(0, config.batchSize);
			} else {
				for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
					fnRunMinibatch(mbs, mbs + config.miniBatchSize);
				}
			}
		}
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = models["policy"]->CopyParams();
	auto criticAfter = models["critic"]->CopyParams();

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();
	float sharedHeadUpdateMagnitude = 0.f;
	if (models["shared_head"]) {
		auto sharedHeadAfter = models["shared_head"]->CopyParams();
		sharedHeadUpdateMagnitude = (sharedHeadBefore - sharedHeadAfter).norm().item<float>();
	}

	// Assemble and return report (single GPU sync here when useGpuAccum)
	if (useGpuAccum && tCount > 0) {
		auto tEntropy = (tEntropyCount > 0) ? (tEntropySum / tEntropyCount).cpu() : torch::tensor(0.f);
		auto tDiv = (tDivergenceSum / tCount).cpu();
		auto tPol = (tPolicyLossSum / tCount).cpu();
		auto tCrit = (tCriticLossSum / tCount).cpu();
		auto tClipVal = (tClipSum / tCount).cpu();
		report["Policy Entropy"] = tEntropy.item<float>();
		report["Mean KL Divergence"] = tDiv.item<float>();
		if (!isFirstIteration) {
			report["Policy Loss"] = tPol.item<float>();
			report["Critic Loss"] = tCrit.item<float>();
			if (config.useGuidingPolicy)
				report["Guiding Loss"] = (tGuidingLossSum / tCount).cpu().item<float>();
			report["SB3 Clip Fraction"] = tClipVal.item<float>();
			report["Policy Update Magnitude"] = policyUpdateMagnitude;
			report["Critic Update Magnitude"] = criticUpdateMagnitude;
			if (models["shared_head"])
				report["Shared Head Update Magnitude"] = sharedHeadUpdateMagnitude;
		}
	} else {
		report["Policy Entropy"] = avgEntropy.Get();
		report["Mean KL Divergence"] = avgDivergence.Get();
		if (!isFirstIteration) {
			report["Policy Loss"] = avgPolicyLoss.Get();
			report["Critic Loss"] = avgCriticLoss.Get();
			if (config.useGuidingPolicy)
				report["Guiding Loss"] = avgGuidingLoss.Get();
			report["SB3 Clip Fraction"] = avgClip.Get();
			report["Policy Update Magnitude"] = policyUpdateMagnitude;
			report["Critic Update Magnitude"] = criticUpdateMagnitude;
			if (models["shared_head"])
				report["Shared Head Update Magnitude"] = sharedHeadUpdateMagnitude;
		}
	}
}

void GGL::PPOLearner::TransferLearn(
	ModelSet& oldModels,
	torch::Tensor newObs, torch::Tensor oldObs,
	torch::Tensor newActionMasks, torch::Tensor oldActionMasks,
	torch::Tensor actionMaps,
	Report& report,
	const TransferLearnConfig& tlConfig
) {

	torch::Tensor oldProbs;
	{ // No grad for old model inference
		RG_NO_GRAD;
		oldProbs = InferPolicyProbsFromModels(oldModels, oldObs, oldActionMasks, config.policyTemperature, config.useHalfPrecision);
		report["Old Policy Entropy"] = ComputeEntropy(oldProbs, oldActionMasks, config.maskEntropy).detach().cpu().item<float>();

		if (actionMaps.defined())
			oldProbs = oldProbs.gather(1, actionMaps);
	}

	for (auto& model : GetPolicyModels())
		model->SetOptimLR(tlConfig.lr);

	auto policyBefore = models["policy"]->CopyParams();
	
	for (int i = 0; i < tlConfig.epochs; i++) {
		torch::Tensor newProbs = InferPolicyProbsFromModels(models, newObs, newActionMasks, config.policyTemperature, false);

		// Non-summative KL div	loss
		torch::Tensor transferLearnLoss;
		if (tlConfig.useKLDiv) {
			transferLearnLoss = (oldProbs * torch::log(oldProbs / newProbs)).abs();
		} else {
			transferLearnLoss = (oldProbs - newProbs).abs();
		}
		transferLearnLoss = transferLearnLoss.pow(tlConfig.lossExponent);
		transferLearnLoss = transferLearnLoss.mean();
		transferLearnLoss *= tlConfig.lossScale;

		if (i == 0) {
			RG_NO_GRAD;
			torch::Tensor matchingActionsMask = (newProbs.detach().argmax(-1) == oldProbs.detach().argmax(-1));
			report["Transfer Learn Accuracy"] = matchingActionsMask.to(torch::kFloat).mean().cpu().item<float>();
			report["Transfer Learn Loss"] = transferLearnLoss.detach().cpu().item<float>();

			report["Policy Entropy"] = ComputeEntropy(newProbs, newActionMasks, config.maskEntropy).detach().cpu().item<float>();
		}

		transferLearnLoss.backward();

		models.StepOptims();
	}

	auto policyAfter = models["policy"]->CopyParams();
	report["Policy Update Magnitude"] = (policyBefore - policyAfter).norm().item<float>();
}

void GGL::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	models.Save(folderPath);
}

void GGL::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	models.Load(folderPath, true, true);

	SetLearningRates(config.policyLR, config.criticLR);
}

void GGL::PPOLearner::SetLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	models["policy"]->SetOptimLR(policyLR);
	models["critic"]->SetOptimLR(criticLR);

	if (models["shared_head"])
		models["shared_head"]->SetOptimLR(RS_MIN(policyLR, criticLR));

	RG_LOG("PPOLearner: " << RS_STR(std::scientific << "Set learning rate to [" << policyLR << ", " << criticLR << "]"));
}

GGL::ModelSet GGL::PPOLearner::GetPolicyModels() {
	ModelSet result = {};
	for (Model* model : models) {
		if (model->modelName == "critic")
			continue;
		
		result.Add(model);
	}
	return result;
}
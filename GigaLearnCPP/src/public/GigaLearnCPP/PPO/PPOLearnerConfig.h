#pragma once
#include <RLGymCPP/BasicTypes/Lists.h>

#include "../Util/ModelConfig.h"

namespace GGL {

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/ppo_learner.py
	struct PPOLearnerConfig {

		int64_t tsPerItr = 50'000;
		int64_t batchSize = 50'000;
		int64_t miniBatchSize = 0; // Set to 0 to just use batchSize

		// On the last batch of the iteration, 
		//	if the amount of remaining experience exceeds the batch size, 
		//	all remaining experience is used as a larger batch.
		// This prevents experience loss due to batch size rounding.
		// This will only happen if the amount of remaining experience is < batchSize*2.
		bool overbatching = true;

		double maxEpisodeDuration = 120; // In seconds

		// Actions with the highest probability are always chosen, instead of being more likely
		// This will make your bot play better (usually), but is horrible for learning
		// Trying to run a PPO learn iteration with deterministic mode will throw an exception
		bool deterministic = false;

		// Use half-precision models for inference
		// This is much faster on GPU, not so much for CPU
		bool useHalfPrecision = false;

		PartialModelConfig policy, critic, sharedHead;

		int epochs = 10;  // rocket-learn default
		float policyLR = 3e-4f; // Policy learning rate
		float criticLR = 3e-4f; // Critic learning rate

		float entropyScale = 0.01f;  // rocket-learn ent_coef: raw entropy scale
		bool maskEntropy = false;     // rocket-learn: no action-mask entropy adjustment

		float clipRange = 0.2f;       // rocket-learn clip_range
		float policyTemperature = 1;

		float gaeLambda = 0.95f;      // rocket-learn gae_lambda
		float gaeGamma = 0.99f;       // rocket-learn gamma
		float rewardClipRange = 10; // Clip range for normalized rewards, set 0 to disable

		bool useGuidingPolicy = false;
		std::filesystem::path guidingPolicyPath = "guiding_policy/"; // Path of the guiding policy model(s)
		float guidingStrength = 0.03f;

		PPOLearnerConfig() {
			policy = {};
			policy.layerSizes = { 256, 256, 256 };
			critic = {};
			critic.layerSizes = { 256, 256, 256 };
			sharedHead = {};
			sharedHead.layerSizes = { 256 };
			sharedHead.addOutputLayer = false;
		}
	};
}
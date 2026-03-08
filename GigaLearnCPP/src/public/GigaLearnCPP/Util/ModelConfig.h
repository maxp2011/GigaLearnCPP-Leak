#pragma once

#include "../Framework.h"

namespace GGL {
	enum class ModelOptimType {
		ADAM,
		ADAMW,
		ADAGRAD,
		RMSPROP,
		MAGSGD
	};

	enum class ModelActivationType {
		RELU,
		LEAKY_RELU,
		GELU,
		SILU,   // Swish: x * sigmoid(x)
		SIGMOID,
		TANH
	};

	// Doesn't include inputs or outputs
	struct PartialModelConfig {
		std::vector<int> layerSizes = {};
		ModelActivationType activationType = ModelActivationType::LEAKY_RELU;
		ModelOptimType optimType = ModelOptimType::ADAMW;
		float weightDecay = 0.01f;  // AdamW L2 decoupled weight decay
		bool addLayerNorm = true;
		bool addOutputLayer = true;

		bool IsValid() const {
			return !layerSizes.empty();
		}
	};

	struct ModelConfig : PartialModelConfig {
		int numInputs = -1;
		int numOutputs = -1;

		bool IsValid() const {
			return PartialModelConfig::IsValid() && numInputs > 0 && (numOutputs > 0 || !addOutputLayer);
		}

		ModelConfig(const PartialModelConfig& partialConfig) : PartialModelConfig(partialConfig) {}
	};
}
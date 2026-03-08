#include "../../src/RLBotClient.h"

#include <RLGymCPP/ObsBuilders/CustomObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <GigaLearnCPP/Util/InferUnit.h>

#include <rlbot/platform.h>

#include <cstring>
#include <filesystem>

using namespace GGL;
using namespace RLGC;

static std::string ParseStrArg(int argc, char* argv[], const char* flag, const char* defaultValue) {
	for (int i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], flag) == 0)
			return argv[i + 1];
	}
	return defaultValue ? std::string(defaultValue) : std::string();
}

int main(int argc, char* argv[]) {
	std::string exeDir = rlbot::platform::GetExecutableDirectory();
	rlbot::platform::SetWorkingDirectory(exeDir);

	std::string meshPath = "collision_meshes";
	std::string modelsPath = "model_checkpoint";

	// Support positional args: exe <collision_meshes> <model_path>
	// Also support named: --meshes <path> --models <path>
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-dll-path") == 0) { i++; continue; }
		if (strcmp(argv[i], "--meshes") == 0 && i + 1 < argc) { meshPath = argv[++i]; continue; }
		if (strcmp(argv[i], "--models") == 0 && i + 1 < argc) { modelsPath = argv[++i]; continue; }
		if (argv[i][0] != '-') {
			if (meshPath == "collision_meshes") meshPath = argv[i];
			else modelsPath = argv[i];
		}
	}

	RocketSim::Init(meshPath);

	constexpr int OBS_SIZE = 323;
	constexpr int TICK_SKIP = 6;
	constexpr int ACTION_DELAY = TICK_SKIP - 1;

	PartialModelConfig sharedHeadConfig = {};
	sharedHeadConfig.layerSizes = { 1024, 1024, 1024, 1024, 512 };
	sharedHeadConfig.activationType = ModelActivationType::LEAKY_RELU;
	sharedHeadConfig.addLayerNorm = false;
	sharedHeadConfig.addOutputLayer = false;

	PartialModelConfig policyConfig = {};
	policyConfig.layerSizes = { 1024, 1024, 1024, 1024, 512 };
	policyConfig.activationType = ModelActivationType::LEAKY_RELU;
	policyConfig.addLayerNorm = false;

	bool useGPU = true;

	InferUnit* inferUnit = new InferUnit(
		new CustomObs(),
		OBS_SIZE,
		new DefaultAction(),
		sharedHeadConfig,
		policyConfig,
		modelsPath,
		useGPU
	);

	RLBotParams params = {};
	params.port = 42653;
	params.tickSkip = TICK_SKIP;
	params.actionDelay = ACTION_DELAY;
	params.inferUnit = inferUnit;

	RLBotClient::Run(params);

	return EXIT_SUCCESS;
}

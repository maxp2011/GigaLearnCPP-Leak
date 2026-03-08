#pragma once

#include <cstdint>
#include <string>

namespace GGL {

	// Distributed training: use multiple PCs for experience collection.
	// One machine runs as --learner (GPU, runs PPO), others as --worker (collect experience).
	struct DistributedConfig {
		bool enabled = false;
		bool isLearner = false;   // This process is the central learner
		bool isWorker = false;    // This process is a worker collecting experience

		std::string learnerHost = "127.0.0.1";
		uint16_t learnerPort = 29500;

		// Learner: also run local collection (0 = workers only, >0 = local games too)
		int localNumGames = 0;

		// How often learner sends updated weights to workers (iterations)
		int weightSyncInterval = 1;

		// Worker: path to load initial weights before first sync (optional)
		std::string initialWeightsPath = "";

		// Redis mode (scales to 3000+ workers): when set, use Redis instead of TCP
		std::string redisHost = "";
		int redisPort = 6379;
	};
}

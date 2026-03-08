#pragma once

#include <RLGymCPP/EnvSet/EnvSet.h>
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include "PPO/TransferLearnConfig.h"

namespace GGL {

	typedef std::function<void(class Learner*, const std::vector<RLGC::GameState>& states, Report& report)> StepCallbackFn;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		LearnerConfig config;

		RLGC::EnvSet* envSet;

		class PPOLearner* ppo;
		class PolicyVersionManager* versionMgr;

		RLGC::EnvCreateFn envCreateFn;
		MetricSender* metricSender;
		RenderSender* renderSender;

		int obsSize;
		int numActions;

		struct WelfordStat* returnStat;
		struct WelfordStat* rewardStat;   // rocket-learn: running mean/var for reward normalization
		struct BatchedWelfordStat* obsStat;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalIterations = 0;

		StepCallbackFn stepCallback = NULL;

		Learner(RLGC::EnvCreateFn envCreateFunc, LearnerConfig config, StepCallbackFn stepCallback = NULL);
		void Start();
		void StartDistributedLearner();
		void StartDistributedWorker();
#ifdef GIGL_REDIS
		void StartDistributedLearnerRedis();
		void StartDistributedWorkerRedis();
#endif
		void StartTransferLearn(const TransferLearnConfig& transferLearnConfig);

		void StartQuitKeyThread(bool& quitPressed, std::thread& outThread);

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		RG_NO_COPY(Learner);

		~Learner();
	};
}
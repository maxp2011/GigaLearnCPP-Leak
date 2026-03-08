#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
#include <GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <GigaLearnCPP/Distributed/DistributedTransport.h>
#include <GigaLearnCPP/Distributed/DistributedConfig.h>
#ifdef GIGL_REDIS
#include <GigaLearnCPP/Distributed/RedisTransport.h>
#endif

#include <private/GigaLearnCPP/PPO/GAE.h>
#include <private/GigaLearnCPP/PPO/ExperienceBuffer.h>
#include <private/GigaLearnCPP/Distributed/DistributedProtocol.h>
#include <private/GigaLearnCPP/FrameworkTorch.h>
#include <private/GigaLearnCPP/Util/WelfordStat.h>

#include <RLGymCPP/BasicTypes/Lists.h>
#include <fstream>
#include <thread>
#include <atomic>

using namespace RLGC;

namespace {

struct Trajectory {
	FList states, nextStates, rewards, logProbs;
	std::vector<uint8_t> actionMasks;
	std::vector<int8_t> terminals;
	std::vector<int32_t> actions;

	void Append(const GGL::DistributedTrajectory& other) {
		states.insert(states.end(), other.states.begin(), other.states.end());
		nextStates.insert(nextStates.end(), other.nextStates.begin(), other.nextStates.end());
		rewards.insert(rewards.end(), other.rewards.begin(), other.rewards.end());
		logProbs.insert(logProbs.end(), other.logProbs.begin(), other.logProbs.end());
		actionMasks.insert(actionMasks.end(), other.actionMasks.begin(), other.actionMasks.end());
		terminals.insert(terminals.end(), other.terminals.begin(), other.terminals.end());
		actions.insert(actions.end(), other.actions.begin(), other.actions.end());
	}

	size_t Length() const { return actions.size(); }
};

} // namespace

void GGL::Learner::StartDistributedLearner() {
	auto& d = config.distributed;

#ifdef GIGL_REDIS
	if (!d.redisHost.empty()) {
		StartDistributedLearnerRedis();
		return;
	}
#endif

	GGL::DistributedTransport transport;
	if (!transport.StartServer(d.learnerPort)) {
		RG_ERR_CLOSE("Distributed: Failed to start server on port " << d.learnerPort);
	}
	RG_LOG("Distributed learner: listening on port " << d.learnerPort << ", waiting for workers...");

	// Block until at least one worker connects
	int clientId = transport.AcceptClient();
	if (clientId < 0) {
		RG_ERR_CLOSE("Distributed: Accept failed");
	}
	RG_LOG("Distributed: Worker connected (client " << clientId << ")");

	// Send config to worker (obsSize, numActions)
	{
		uint32_t buf[3] = { (uint32_t)DistMsgType::CONFIG, (uint32_t)obsSize, (uint32_t)numActions };
		if (!transport.SendToClient(clientId, buf, sizeof(buf))) {
			RG_ERR_CLOSE("Distributed: Failed to send config");
		}
	}

	// Send initial weights
	std::filesystem::path weightDir = std::filesystem::temp_directory_path() / "gigl_dist_learner";
	std::filesystem::create_directories(weightDir);
	ppo->SaveTo(weightDir);

	std::vector<uint8_t> weightBlob;
	for (auto& entry : std::filesystem::directory_iterator(weightDir)) {
		if (!entry.is_regular_file()) continue;
		std::ifstream f(entry.path(), std::ios::binary);
		f.seekg(0, std::ios::end);
		size_t sz = f.tellg();
		f.seekg(0);
		size_t off = weightBlob.size();
		weightBlob.resize(off + 4 + entry.path().filename().string().size() + 4 + sz);
		uint32_t nameLen = (uint32_t)entry.path().filename().string().size();
		memcpy(weightBlob.data() + off, &nameLen, 4);
		memcpy(weightBlob.data() + off + 4, entry.path().filename().string().c_str(), nameLen);
		uint32_t fileLen = (uint32_t)sz;
		memcpy(weightBlob.data() + off + 4 + nameLen, &fileLen, 4);
		f.read((char*)(weightBlob.data() + off + 4 + nameLen + 4), sz);
	}

	uint32_t msgType = (uint32_t)DistMsgType::WEIGHTS;
	uint32_t totalLen = (uint32_t)weightBlob.size();
	std::vector<uint8_t> sendBuf;
	sendBuf.resize(4 + 4 + weightBlob.size());
	memcpy(sendBuf.data(), &msgType, 4);
	memcpy(sendBuf.data() + 4, &totalLen, 4);
	memcpy(sendBuf.data() + 8, weightBlob.data(), weightBlob.size());
	if (!transport.SendToClient(clientId, sendBuf.data(), sendBuf.size())) {
		RG_ERR_CLOSE("Distributed: Failed to send weights");
	}
	RG_LOG("Distributed: Sent initial weights to worker");

	ExperienceBuffer experience(config.randomSeed, ppo->device);
	Trajectory combinedTraj;
	int64_t tsPerItr = config.ppo.tsPerItr;
	bool saveQueued = false;
	std::thread keyPressThread;
	StartQuitKeyThread(saveQueued, keyPressThread);

	while (true) {
		Report report = {};
		bool isFirstIteration = (totalTimesteps == 0);

		// Receive trajectories from worker until we have enough
		combinedTraj = Trajectory{};
		while ((int64_t)combinedTraj.Length() < tsPerItr) {
			std::vector<uint8_t> recvData;
			int from = transport.ReceiveFromClient(recvData);
			if (from < 0) {
				RG_ERR_CLOSE("Distributed: Worker disconnected");
			}
			GGL::DistributedTrajectory dt;
			if (recvData.size() >= 20 && GGL::DeserializeTrajectory(recvData.data(), recvData.size(), obsSize, numActions, dt)) {
				combinedTraj.Append(dt);
			}
		}

		int stepsCollected = (int)combinedTraj.Length();
		totalTimesteps += stepsCollected;

		// Process: same as normal Learner
		RG_NO_GRAD;

		torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
		torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
		torch::Tensor tActions = torch::tensor(combinedTraj.actions);
		torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
		torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
		torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);

		torch::Tensor tNextTruncStates;
		if (!combinedTraj.nextStates.empty())
			tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });

		auto dev = ppo->device;
		torch::Tensor tValPreds, tTruncValPreds;

		if (dev.is_cpu()) {
			tValPreds = ppo->InferCritic(tStates.to(dev, true, true));
			if (tNextTruncStates.defined())
				tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(dev, true, true));
		} else {
			tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() }, dev);
			for (int i = 0; i < (int)combinedTraj.Length(); i += config.ppo.miniBatchSize) {
				int end = RS_MIN(i + config.ppo.miniBatchSize, (int)combinedTraj.Length());
				auto part = ppo->InferCritic(tStates.slice(0, i, end).to(dev, true, true));
				tValPreds.slice(0, i, end).copy_(part, true);
			}
			if (tNextTruncStates.defined())
				tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(dev, true, true));
			tValPreds = tValPreds.cpu();
			if (tTruncValPreds.defined())
				tTruncValPreds = tTruncValPreds.cpu();
		}

		if (rewardStat) {
			rewardStat->Increment(TENSOR_TO_VEC<float>(tRewards));
			float rMean = (float)rewardStat->GetMean();
			float rStd = (float)rewardStat->GetSTD() + 1e-8f;
			tRewards = (tRewards - rMean) / rStd;
			if (config.ppo.rewardClipRange > 0)
				tRewards = torch::clamp(tRewards, -config.ppo.rewardClipRange, config.ppo.rewardClipRange);
		}

		torch::Tensor tAdvantages, tTargetVals, tReturns;
		float rewClipPortion;
		GAE::Compute(tRewards, tTerminals, tValPreds, tTruncValPreds,
			tAdvantages, tTargetVals, tReturns, rewClipPortion,
			config.ppo.gaeGamma, config.ppo.gaeLambda,
			rewardStat ? 0 : (returnStat ? (float)returnStat->GetSTD() : 1),
			rewardStat ? 0 : config.ppo.rewardClipRange);

		experience.data.states = tStates.to(dev);
		experience.data.actions = tActions.to(dev);
		experience.data.logProbs = tLogProbs.to(dev);
		experience.data.actionMasks = tActionMasks.to(dev);
		experience.data.advantages = tAdvantages.to(dev);
		experience.data.targetValues = tTargetVals.to(dev);

		ppo->Learn(experience, report, isFirstIteration);
		totalIterations++;

		report["Collected Timesteps"] = stepsCollected;
		report["Total Timesteps"] = totalTimesteps;
		report["Total Iterations"] = totalIterations;

		// Send weights or ACK to worker (worker blocks on receive after each send)
		if (totalIterations % d.weightSyncInterval == 0) {
			ppo->SaveTo(weightDir);
			weightBlob.clear();
			for (auto& entry : std::filesystem::directory_iterator(weightDir)) {
				if (!entry.is_regular_file()) continue;
				std::ifstream f(entry.path(), std::ios::binary);
				f.seekg(0, std::ios::end);
				size_t sz = f.tellg();
				f.seekg(0);
				size_t off = weightBlob.size();
				weightBlob.resize(off + 4 + entry.path().filename().string().size() + 4 + sz);
				uint32_t nameLen = (uint32_t)entry.path().filename().string().size();
				memcpy(weightBlob.data() + off, &nameLen, 4);
				memcpy(weightBlob.data() + off + 4, entry.path().filename().string().c_str(), nameLen);
				uint32_t fileLen = (uint32_t)sz;
				memcpy(weightBlob.data() + off + 4 + nameLen, &fileLen, 4);
				f.read((char*)(weightBlob.data() + off + 4 + nameLen + 4), sz);
			}
			sendBuf.resize(8 + weightBlob.size());
			msgType = (uint32_t)DistMsgType::WEIGHTS;
			totalLen = (uint32_t)weightBlob.size();
			memcpy(sendBuf.data(), &msgType, 4);
			memcpy(sendBuf.data() + 4, &totalLen, 4);
			memcpy(sendBuf.data() + 8, weightBlob.data(), weightBlob.size());
			transport.SendToClient(clientId, sendBuf.data(), sendBuf.size());
		} else {
			// ACK so worker can continue
			uint32_t ackMsg = (uint32_t)DistMsgType::ACK;
			uint32_t zero = 0;
			sendBuf.resize(8);
			memcpy(sendBuf.data(), &ackMsg, 4);
			memcpy(sendBuf.data() + 4, &zero, 4);
			transport.SendToClient(clientId, sendBuf.data(), 8);
		}

		if (saveQueued && !config.checkpointFolder.empty()) {
			Save();
			exit(0);
		}
		if (!config.checkpointFolder.empty() && totalTimesteps / config.tsPerSave > (totalTimesteps - stepsCollected) / config.tsPerSave) {
			Save();
		}

		report.Finish();
		if (metricSender)
			metricSender->Send(report);
	}
}

void GGL::Learner::StartDistributedWorker() {
	auto& d = config.distributed;

#ifdef GIGL_REDIS
	if (!d.redisHost.empty()) {
		StartDistributedWorkerRedis();
		return;
	}
#endif

	GGL::DistributedTransport transport;
	if (!transport.Connect(d.learnerHost, d.learnerPort)) {
		RG_ERR_CLOSE("Distributed worker: Failed to connect to " << d.learnerHost << ":" << d.learnerPort);
	}
	RG_LOG("Distributed worker: connected to learner");

	// Receive config
	std::vector<uint8_t> recvData;
	if (!transport.Receive(recvData) || recvData.size() < 12) {
		RG_ERR_CLOSE("Distributed worker: Failed to receive config");
	}
	uint32_t* p = (uint32_t*)recvData.data();
	if (p[0] != (uint32_t)DistMsgType::CONFIG) {
		RG_ERR_CLOSE("Distributed worker: Expected CONFIG message");
	}
	int rObsSize = (int)p[1];
	int rNumActions = (int)p[2];
	if (rObsSize != obsSize || rNumActions != numActions) {
		RG_ERR_CLOSE("Distributed worker: Config mismatch (obs " << rObsSize << " vs " << obsSize << ", acts " << rNumActions << " vs " << numActions << ")");
	}

	// Receive initial weights
	if (!transport.Receive(recvData) || recvData.size() < 8) {
		RG_ERR_CLOSE("Distributed worker: Failed to receive weights");
	}
	p = (uint32_t*)recvData.data();
	if (p[0] != (uint32_t)DistMsgType::WEIGHTS) {
		RG_ERR_CLOSE("Distributed worker: Expected WEIGHTS message");
	}
	std::filesystem::path weightDir = std::filesystem::temp_directory_path() / "gigl_dist_worker";
	std::filesystem::create_directories(weightDir);
	const uint8_t* data = recvData.data() + 8;
	size_t remaining = recvData.size() - 8;
	while (remaining >= 8) {
		uint32_t nameLen = *(uint32_t*)data;
		uint32_t fileLen = *(uint32_t*)(data + 4 + nameLen);
		if (remaining < 8 + nameLen + fileLen) break;
		std::string fname((char*)(data + 4), nameLen);
		std::ofstream f(weightDir / fname, std::ios::binary);
		f.write((char*)(data + 8 + nameLen), fileLen);
		data += 8 + nameLen + fileLen;
		remaining -= 8 + nameLen + fileLen;
	}
	ppo->LoadFrom(weightDir);
	RG_LOG("Distributed worker: loaded initial weights");

	// Worker loop: collect and send trajectories
	int numPlayers = envSet->state.numPlayers;
	int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

	struct Trajectory {
		FList states, nextStates, rewards, logProbs;
		std::vector<uint8_t> actionMasks;
		std::vector<int8_t> terminals;
		std::vector<int32_t> actions;

		void Clear() { states.clear(); nextStates.clear(); rewards.clear(); logProbs.clear(); actionMasks.clear(); terminals.clear(); actions.clear(); }
		size_t Length() const { return actions.size(); }
	};

	auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
	Trajectory combinedTraj;

	while (true) {
		combinedTraj.Clear();
		envSet->Reset();

		while ((int)combinedTraj.Length() < config.ppo.tsPerItr) {
			for (int i = 0; i < numPlayers; i++) {
				trajectories[i].states += envSet->state.obs.GetRow(i);
				trajectories[i].actionMasks += envSet->state.actionMasks.GetRow(i);
			}

			envSet->StepFirstHalf(true);

			torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
			torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);
			torch::Tensor tActions, tLogProbs;
			ppo->InferActions(tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), &tActions, &tLogProbs, nullptr, false);

			auto curActions = TENSOR_TO_VEC<int>(tActions);
			FList newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);

			envSet->Sync();
			envSet->StepSecondHalf(curActions, false);

			for (int i = 0; i < numPlayers; i++) {
				trajectories[i].actions.push_back(curActions[i]);
				trajectories[i].rewards += envSet->state.rewards[i];
				trajectories[i].logProbs.push_back(newLogProbs[i]);
			}

			std::vector<uint8_t> curTerminals(numPlayers, 0);
			for (size_t idx = 0; idx < envSet->arenas.size(); idx++) {
				uint8_t term = envSet->state.terminals[idx];
				if (!term) continue;
				auto startIdx = envSet->state.arenaPlayerStartIdx[idx];
				int playersInArena = (int)envSet->state.gameStates[idx].players.size();
				for (int i = 0; i < playersInArena; i++)
					curTerminals[startIdx + i] = term;
			}

			for (int i = 0; i < numPlayers; i++) {
				uint8_t term = curTerminals[i];
				auto& traj = trajectories[i];
				if (!term && traj.Length() >= maxEpisodeLength)
					term = RLGC::TerminalType::TRUNCATED;

				traj.terminals.push_back((int8_t)term);
				if (term) {
					if (term == RLGC::TerminalType::TRUNCATED)
						traj.nextStates += envSet->state.obs.GetRow(i);
					combinedTraj.states.insert(combinedTraj.states.end(), traj.states.begin(), traj.states.end());
					combinedTraj.nextStates.insert(combinedTraj.nextStates.end(), traj.nextStates.begin(), traj.nextStates.end());
					combinedTraj.rewards.insert(combinedTraj.rewards.end(), traj.rewards.begin(), traj.rewards.end());
					combinedTraj.logProbs.insert(combinedTraj.logProbs.end(), traj.logProbs.begin(), traj.logProbs.end());
					combinedTraj.actionMasks.insert(combinedTraj.actionMasks.end(), traj.actionMasks.begin(), traj.actionMasks.end());
					combinedTraj.terminals.insert(combinedTraj.terminals.end(), traj.terminals.begin(), traj.terminals.end());
					combinedTraj.actions.insert(combinedTraj.actions.end(), traj.actions.begin(), traj.actions.end());
					traj.Clear();
				}
			}
		}

		// Send trajectory to learner
		GGL::DistributedTrajectory dt;
		dt.states = combinedTraj.states;
		dt.nextStates = combinedTraj.nextStates;
		dt.rewards = combinedTraj.rewards;
		dt.logProbs = combinedTraj.logProbs;
		dt.actionMasks = combinedTraj.actionMasks;
		dt.terminals = combinedTraj.terminals;
		dt.actions = combinedTraj.actions;

		std::vector<uint8_t> serialized;
		GGL::SerializeTrajectory(dt, obsSize, numActions, serialized);
		if (!transport.Send(serialized.data(), serialized.size())) {
			RG_ERR_CLOSE("Distributed worker: Failed to send trajectory");
		}

		// Receive weights or ACK from learner (learner always sends after processing)
		if (!transport.Receive(recvData) || recvData.size() < 8) {
			RG_ERR_CLOSE("Distributed worker: Failed to receive response from learner");
		}
		p = (uint32_t*)recvData.data();
		if (p[0] == (uint32_t)DistMsgType::WEIGHTS) {
			const uint8_t* data = recvData.data() + 8;
			size_t remaining = recvData.size() - 8;
			std::filesystem::create_directories(weightDir);
			while (remaining >= 8) {
				uint32_t nameLen = *(uint32_t*)data;
				if (nameLen > 256 || remaining < 8 + nameLen + 4) break;
				uint32_t fileLen = *(uint32_t*)(data + 4 + nameLen);
				if (remaining < 8 + nameLen + fileLen) break;
				std::string fname((char*)(data + 4), nameLen);
				std::ofstream f(weightDir / fname, std::ios::binary);
				f.write((char*)(data + 8 + nameLen), fileLen);
				data += 8 + nameLen + fileLen;
				remaining -= 8 + nameLen + fileLen;
			}
			ppo->LoadFrom(weightDir);
		}
		// ACK: nothing to do, continue with current weights
	}
}

#ifdef GIGL_REDIS
void GGL::Learner::StartDistributedLearnerRedis() {
	auto& d = config.distributed;
	GGL::RedisTransport redis;
	if (!redis.Connect(d.redisHost, d.redisPort)) {
		RG_ERR_CLOSE("Redis: Failed to connect to " << d.redisHost << ":" << d.redisPort);
	}
	RG_LOG("Redis learner: connected, waiting for rollouts...");

	std::filesystem::path weightDir = std::filesystem::temp_directory_path() / "gigl_redis_learner";
	std::filesystem::create_directories(weightDir);
	ppo->SaveTo(weightDir);

	std::vector<uint8_t> weightBlob;
	for (auto& entry : std::filesystem::directory_iterator(weightDir)) {
		if (!entry.is_regular_file()) continue;
		std::ifstream f(entry.path(), std::ios::binary);
		f.seekg(0, std::ios::end);
		size_t sz = f.tellg();
		f.seekg(0);
		size_t off = weightBlob.size();
		weightBlob.resize(off + 4 + entry.path().filename().string().size() + 4 + sz);
		uint32_t nameLen = (uint32_t)entry.path().filename().string().size();
		memcpy(weightBlob.data() + off, &nameLen, 4);
		memcpy(weightBlob.data() + off + 4, entry.path().filename().string().c_str(), nameLen);
		uint32_t fileLen = (uint32_t)sz;
		memcpy(weightBlob.data() + off + 4 + nameLen, &fileLen, 4);
		f.read((char*)(weightBlob.data() + off + 4 + nameLen + 4), sz);
	}
	uint32_t msgType = (uint32_t)DistMsgType::WEIGHTS;
	uint32_t totalLen = (uint32_t)weightBlob.size();
	std::vector<uint8_t> modelBuf(8 + weightBlob.size());
	memcpy(modelBuf.data(), &msgType, 4);
	memcpy(modelBuf.data() + 4, &totalLen, 4);
	memcpy(modelBuf.data() + 8, weightBlob.data(), weightBlob.size());
	redis.SetModel(modelBuf.data(), modelBuf.size());
	RG_LOG("Redis: Initial model published");

	ExperienceBuffer experience(config.randomSeed, ppo->device);
	Trajectory combinedTraj;
	int64_t tsPerItr = config.ppo.tsPerItr;
	bool saveQueued = false;
	std::thread keyPressThread;
	StartQuitKeyThread(saveQueued, keyPressThread);

	while (true) {
		Report report = {};
		bool isFirstIteration = (totalTimesteps == 0);

		combinedTraj = Trajectory{};
		while ((int64_t)combinedTraj.Length() < tsPerItr) {
			std::vector<uint8_t> recvData;
			if (!redis.BlpopRollout(recvData, 0)) continue;
			GGL::DistributedTrajectory dt;
			if (recvData.size() >= 20 && GGL::DeserializeTrajectory(recvData.data(), recvData.size(), obsSize, numActions, dt)) {
				combinedTraj.Append(dt);
			}
		}

		int stepsCollected = (int)combinedTraj.Length();
		totalTimesteps += stepsCollected;

		RG_NO_GRAD;
		torch::Tensor tStates = torch::tensor(combinedTraj.states).reshape({ -1, obsSize });
		torch::Tensor tActionMasks = torch::tensor(combinedTraj.actionMasks).reshape({ -1, numActions });
		torch::Tensor tActions = torch::tensor(combinedTraj.actions);
		torch::Tensor tLogProbs = torch::tensor(combinedTraj.logProbs);
		torch::Tensor tRewards = torch::tensor(combinedTraj.rewards);
		torch::Tensor tTerminals = torch::tensor(combinedTraj.terminals);
		torch::Tensor tNextTruncStates;
		if (!combinedTraj.nextStates.empty())
			tNextTruncStates = torch::tensor(combinedTraj.nextStates).reshape({ -1, obsSize });

		auto dev = ppo->device;
		torch::Tensor tValPreds, tTruncValPreds;
		if (dev.is_cpu()) {
			tValPreds = ppo->InferCritic(tStates.to(dev, true, true));
			if (tNextTruncStates.defined())
				tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(dev, true, true));
		} else {
			tValPreds = torch::zeros({ (int64_t)combinedTraj.Length() }, dev);
			for (int i = 0; i < (int)combinedTraj.Length(); i += config.ppo.miniBatchSize) {
				int end = RS_MIN(i + config.ppo.miniBatchSize, (int)combinedTraj.Length());
				auto part = ppo->InferCritic(tStates.slice(0, i, end).to(dev, true, true));
				tValPreds.slice(0, i, end).copy_(part, true);
			}
			if (tNextTruncStates.defined())
				tTruncValPreds = ppo->InferCritic(tNextTruncStates.to(dev, true, true));
			tValPreds = tValPreds.cpu();
			if (tTruncValPreds.defined())
				tTruncValPreds = tTruncValPreds.cpu();
		}

		if (rewardStat) {
			rewardStat->Increment(TENSOR_TO_VEC<float>(tRewards));
			float rMean = (float)rewardStat->GetMean();
			float rStd = (float)rewardStat->GetSTD() + 1e-8f;
			tRewards = (tRewards - rMean) / rStd;
			if (config.ppo.rewardClipRange > 0)
				tRewards = torch::clamp(tRewards, -config.ppo.rewardClipRange, config.ppo.rewardClipRange);
		}

		torch::Tensor tAdvantages, tTargetVals, tReturns;
		float rewClipPortion;
		GAE::Compute(tRewards, tTerminals, tValPreds, tTruncValPreds,
			tAdvantages, tTargetVals, tReturns, rewClipPortion,
			config.ppo.gaeGamma, config.ppo.gaeLambda,
			rewardStat ? 0 : (returnStat ? (float)returnStat->GetSTD() : 1),
			rewardStat ? 0 : config.ppo.rewardClipRange);

		experience.data.states = tStates.to(dev);
		experience.data.actions = tActions.to(dev);
		experience.data.logProbs = tLogProbs.to(dev);
		experience.data.actionMasks = tActionMasks.to(dev);
		experience.data.advantages = tAdvantages.to(dev);
		experience.data.targetValues = tTargetVals.to(dev);

		ppo->Learn(experience, report, isFirstIteration);
		totalIterations++;

		report["Collected Timesteps"] = stepsCollected;
		report["Total Timesteps"] = totalTimesteps;
		report["Total Iterations"] = totalIterations;

		if (totalIterations % d.weightSyncInterval == 0) {
			ppo->SaveTo(weightDir);
			weightBlob.clear();
			for (auto& entry : std::filesystem::directory_iterator(weightDir)) {
				if (!entry.is_regular_file()) continue;
				std::ifstream f(entry.path(), std::ios::binary);
				f.seekg(0, std::ios::end);
				size_t sz = f.tellg();
				f.seekg(0);
				size_t off = weightBlob.size();
				weightBlob.resize(off + 4 + entry.path().filename().string().size() + 4 + sz);
				uint32_t nameLen = (uint32_t)entry.path().filename().string().size();
				memcpy(weightBlob.data() + off, &nameLen, 4);
				memcpy(weightBlob.data() + off + 4, entry.path().filename().string().c_str(), nameLen);
				uint32_t fileLen = (uint32_t)sz;
				memcpy(weightBlob.data() + off + 4 + nameLen, &fileLen, 4);
				f.read((char*)(weightBlob.data() + off + 4 + nameLen + 4), sz);
			}
			modelBuf.resize(8 + weightBlob.size());
			msgType = (uint32_t)DistMsgType::WEIGHTS;
			totalLen = (uint32_t)weightBlob.size();
			memcpy(modelBuf.data(), &msgType, 4);
			memcpy(modelBuf.data() + 4, &totalLen, 4);
			memcpy(modelBuf.data() + 8, weightBlob.data(), weightBlob.size());
			redis.SetModel(modelBuf.data(), modelBuf.size());
		}

		if (saveQueued && !config.checkpointFolder.empty()) { Save(); exit(0); }
		if (!config.checkpointFolder.empty() && totalTimesteps / config.tsPerSave > (totalTimesteps - stepsCollected) / config.tsPerSave)
			Save();

		report.Finish();
		if (metricSender) metricSender->Send(report);
	}
}

void GGL::Learner::StartDistributedWorkerRedis() {
	auto& d = config.distributed;
	GGL::RedisTransport redis;
	if (!redis.Connect(d.redisHost, d.redisPort)) {
		RG_ERR_CLOSE("Redis worker: Failed to connect to " << d.redisHost << ":" << d.redisPort);
	}
	RG_LOG("Redis worker: connected, waiting for initial model...");

	std::vector<uint8_t> modelData;
	while (!redis.GetModel(modelData) || modelData.size() < 8) {
		RG_SLEEP(1000);
	}
	RG_LOG("Redis worker: got initial model");

	std::filesystem::path weightDir = std::filesystem::temp_directory_path() / "gigl_redis_worker";
	std::filesystem::create_directories(weightDir);
	const uint8_t* data = modelData.data() + 8;
	size_t remaining = modelData.size() - 8;
	while (remaining >= 8) {
		uint32_t nameLen = *(uint32_t*)data;
		if (nameLen > 256 || remaining < 8 + nameLen + 4) break;
		uint32_t fileLen = *(uint32_t*)(data + 4 + nameLen);
		if (remaining < 8 + nameLen + fileLen) break;
		std::string fname((char*)(data + 4), nameLen);
		std::ofstream f(weightDir / fname, std::ios::binary);
		f.write((char*)(data + 8 + nameLen), fileLen);
		data += 8 + nameLen + fileLen;
		remaining -= 8 + nameLen + fileLen;
	}
	ppo->LoadFrom(weightDir);

	int numPlayers = envSet->state.numPlayers;
	int maxEpisodeLength = (int)(config.ppo.maxEpisodeDuration * (120.f / config.tickSkip));

	struct Trajectory {
		FList states, nextStates, rewards, logProbs;
		std::vector<uint8_t> actionMasks;
		std::vector<int8_t> terminals;
		std::vector<int32_t> actions;
		void Clear() { states.clear(); nextStates.clear(); rewards.clear(); logProbs.clear(); actionMasks.clear(); terminals.clear(); actions.clear(); }
		size_t Length() const { return actions.size(); }
	};

	auto trajectories = std::vector<Trajectory>(numPlayers, Trajectory{});
	Trajectory combinedTraj;

	while (true) {
		combinedTraj.Clear();
		envSet->Reset();

		while ((int)combinedTraj.Length() < config.ppo.tsPerItr) {
			for (int i = 0; i < numPlayers; i++) {
				trajectories[i].states += envSet->state.obs.GetRow(i);
				trajectories[i].actionMasks += envSet->state.actionMasks.GetRow(i);
			}
			envSet->StepFirstHalf(true);

			torch::Tensor tStates = DIMLIST2_TO_TENSOR<float>(envSet->state.obs);
			torch::Tensor tActionMasks = DIMLIST2_TO_TENSOR<uint8_t>(envSet->state.actionMasks);
			torch::Tensor tActions, tLogProbs;
			ppo->InferActions(tStates.to(ppo->device, true), tActionMasks.to(ppo->device, true), &tActions, &tLogProbs, nullptr, false);

			auto curActions = TENSOR_TO_VEC<int>(tActions);
			FList newLogProbs = TENSOR_TO_VEC<float>(tLogProbs);
			envSet->Sync();
			envSet->StepSecondHalf(curActions, false);

			for (int i = 0; i < numPlayers; i++) {
				trajectories[i].actions.push_back(curActions[i]);
				trajectories[i].rewards += envSet->state.rewards[i];
				trajectories[i].logProbs.push_back(newLogProbs[i]);
			}

			std::vector<uint8_t> curTerminals(numPlayers, 0);
			for (size_t idx = 0; idx < envSet->arenas.size(); idx++) {
				uint8_t term = envSet->state.terminals[idx];
				if (!term) continue;
				auto startIdx = envSet->state.arenaPlayerStartIdx[idx];
				int playersInArena = (int)envSet->state.gameStates[idx].players.size();
				for (int i = 0; i < playersInArena; i++)
					curTerminals[startIdx + i] = term;
			}

			for (int i = 0; i < numPlayers; i++) {
				uint8_t term = curTerminals[i];
				auto& traj = trajectories[i];
				if (!term && traj.Length() >= maxEpisodeLength)
					term = RLGC::TerminalType::TRUNCATED;
				traj.terminals.push_back((int8_t)term);
				if (term) {
					if (term == RLGC::TerminalType::TRUNCATED)
						traj.nextStates += envSet->state.obs.GetRow(i);
					combinedTraj.states.insert(combinedTraj.states.end(), traj.states.begin(), traj.states.end());
					combinedTraj.nextStates.insert(combinedTraj.nextStates.end(), traj.nextStates.begin(), traj.nextStates.end());
					combinedTraj.rewards.insert(combinedTraj.rewards.end(), traj.rewards.begin(), traj.rewards.end());
					combinedTraj.logProbs.insert(combinedTraj.logProbs.end(), traj.logProbs.begin(), traj.logProbs.end());
					combinedTraj.actionMasks.insert(combinedTraj.actionMasks.end(), traj.actionMasks.begin(), traj.actionMasks.end());
					combinedTraj.terminals.insert(combinedTraj.terminals.end(), traj.terminals.begin(), traj.terminals.end());
					combinedTraj.actions.insert(combinedTraj.actions.end(), traj.actions.begin(), traj.actions.end());
					traj.Clear();
				}
			}
		}

		GGL::DistributedTrajectory dt;
		dt.states = combinedTraj.states;
		dt.nextStates = combinedTraj.nextStates;
		dt.rewards = combinedTraj.rewards;
		dt.logProbs = combinedTraj.logProbs;
		dt.actionMasks = combinedTraj.actionMasks;
		dt.terminals = combinedTraj.terminals;
		dt.actions = combinedTraj.actions;

		std::vector<uint8_t> serialized;
		GGL::SerializeTrajectory(dt, obsSize, numActions, serialized);
		redis.RpushRollout(serialized.data(), serialized.size());

		if (redis.GetModel(modelData) && modelData.size() >= 8) {
			uint32_t* p = (uint32_t*)modelData.data();
			if (p[0] == (uint32_t)DistMsgType::WEIGHTS) {
				data = modelData.data() + 8;
				remaining = modelData.size() - 8;
				while (remaining >= 8) {
					uint32_t nameLen = *(uint32_t*)data;
					if (nameLen > 256 || remaining < 8 + nameLen + 4) break;
					uint32_t fileLen = *(uint32_t*)(data + 4 + nameLen);
					if (remaining < 8 + nameLen + fileLen) break;
					std::string fname((char*)(data + 4), nameLen);
					std::ofstream f(weightDir / fname, std::ios::binary);
					f.write((char*)(data + 8 + nameLen), fileLen);
					data += 8 + nameLen + fileLen;
					remaining -= 8 + nameLen + fileLen;
				}
				ppo->LoadFrom(weightDir);
			}
		}
	}
}
#endif

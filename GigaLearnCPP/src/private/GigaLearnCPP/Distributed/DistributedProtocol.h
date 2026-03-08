#pragma once

#include <RLGymCPP/BasicTypes/Lists.h>
#include <vector>
#include <cstdint>

namespace GGL {

	// Trajectory data for network transfer (matches Learner's Trajectory struct)
	struct DistributedTrajectory {
		RLGC::FList states, nextStates, rewards, logProbs;
		std::vector<uint8_t> actionMasks;
		std::vector<int8_t> terminals;
		std::vector<int32_t> actions;
	};

	// Message types
	enum class DistMsgType : uint32_t {
		TRAJECTORY = 1,
		WEIGHTS = 2,
		CONFIG = 3,   // Learner -> Worker: obsSize, numActions
		ACK = 4,
	};

	// Serialize trajectory to bytes (for worker -> learner)
	void SerializeTrajectory(const DistributedTrajectory& t, int obsSize, int numActions, std::vector<uint8_t>& out);

	// Deserialize trajectory from bytes
	bool DeserializeTrajectory(const uint8_t* data, size_t size, int obsSize, int numActions, DistributedTrajectory& out);
}

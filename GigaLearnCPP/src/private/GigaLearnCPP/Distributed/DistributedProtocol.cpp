#include "GigaLearnCPP/Distributed/DistributedProtocol.h"
#include <cstring>
#include <algorithm>

namespace GGL {

static void AppendU32(std::vector<uint8_t>& out, uint32_t v) {
	out.push_back((uint8_t)(v & 0xFF));
	out.push_back((uint8_t)((v >> 8) & 0xFF));
	out.push_back((uint8_t)((v >> 16) & 0xFF));
	out.push_back((uint8_t)((v >> 24) & 0xFF));
}

static uint32_t ReadU32(const uint8_t*& p) {
	uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
	p += 4;
	return v;
}

void SerializeTrajectory(const DistributedTrajectory& t, int obsSize, int numActions, std::vector<uint8_t>& out) {
	out.clear();
	AppendU32(out, (uint32_t)DistMsgType::TRAJECTORY);
	AppendU32(out, (uint32_t)obsSize);
	AppendU32(out, (uint32_t)numActions);

	size_t n = t.actions.size();
	AppendU32(out, (uint32_t)n);

	if (n == 0) return;

	out.insert(out.end(), (uint8_t*)t.states.data(), (uint8_t*)(t.states.data() + t.states.size()));
	out.insert(out.end(), (uint8_t*)t.actions.data(), (uint8_t*)(t.actions.data() + t.actions.size()));
	out.insert(out.end(), (uint8_t*)t.rewards.data(), (uint8_t*)(t.rewards.data() + t.rewards.size()));
	out.insert(out.end(), (uint8_t*)t.logProbs.data(), (uint8_t*)(t.logProbs.data() + t.logProbs.size()));
	out.insert(out.end(), t.actionMasks.data(), t.actionMasks.data() + t.actionMasks.size());
	out.insert(out.end(), (uint8_t*)t.terminals.data(), (uint8_t*)(t.terminals.data() + t.terminals.size()));
	out.insert(out.end(), (uint8_t*)t.nextStates.data(), (uint8_t*)(t.nextStates.data() + t.nextStates.size()));
}

bool DeserializeTrajectory(const uint8_t* data, size_t size, int obsSize, int numActions, DistributedTrajectory& out) {
	if (size < 20) return false;  // type + obsSize + numActions + n
	const uint8_t* p = data;

	uint32_t type = ReadU32(p);
	if (type != (uint32_t)DistMsgType::TRAJECTORY) return false;

	uint32_t rObs = ReadU32(p);
	uint32_t rActs = ReadU32(p);
	uint32_t n = ReadU32(p);

	if (rObs != (uint32_t)obsSize || rActs != (uint32_t)numActions) return false;
	if (n == 0) return true;

	size_t stateBytes = n * obsSize * sizeof(float);
	size_t actionBytes = n * sizeof(int32_t);
	size_t rewardBytes = n * sizeof(float);
	size_t logProbBytes = n * sizeof(float);
	size_t maskBytes = n * numActions * sizeof(uint8_t);
	size_t termBytes = n * sizeof(int8_t);
	size_t totalFixed = 20 + stateBytes + actionBytes + rewardBytes + logProbBytes + maskBytes + termBytes;
	if (size < totalFixed) return false;

	// Parse fixed-size fields first
	out.states.resize(n * obsSize);
	memcpy(out.states.data(), p, stateBytes);
	p += stateBytes;

	out.actions.resize(n);
	memcpy(out.actions.data(), p, actionBytes);
	p += actionBytes;

	out.rewards.resize(n);
	memcpy(out.rewards.data(), p, rewardBytes);
	p += rewardBytes;

	out.logProbs.resize(n);
	memcpy(out.logProbs.data(), p, logProbBytes);
	p += logProbBytes;

	out.actionMasks.resize(n * numActions);
	memcpy(out.actionMasks.data(), p, maskBytes);
	p += maskBytes;

	out.terminals.resize(n);
	memcpy(out.terminals.data(), p, termBytes);
	p += termBytes;

	int truncCount = 0;
	for (int8_t term : out.terminals)
		if (term == 2) truncCount++;  // RLGC::TerminalType::TRUNCATED

	size_t nextStateBytes = (size_t)truncCount * obsSize * sizeof(float);
	if (size < totalFixed + nextStateBytes) return false;

	out.nextStates.resize(truncCount * obsSize);
	if (nextStateBytes > 0)
		memcpy(out.nextStates.data(), p, nextStateBytes);

	return true;
}

}

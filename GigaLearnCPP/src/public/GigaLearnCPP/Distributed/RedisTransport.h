#pragma once

#ifdef GIGL_REDIS

#include <cstdint>
#include <vector>
#include <string>

namespace GGL {

	// Redis-based transport for 3000+ workers (rocket-learn style).
	// Workers RPUSH rollouts, learner BLPOP. Model via GET/SET.
	class RedisTransport {
	public:
		RedisTransport();
		~RedisTransport();

		bool Connect(const std::string& host, int port = 6379);
		void Disconnect();

		// Learner: block until rollout received
		bool BlpopRollout(std::vector<uint8_t>& outData, int timeoutSec = 0);
		bool SetModel(const void* data, size_t size);

		// Worker: push rollout, get latest model (non-blocking)
		bool RpushRollout(const void* data, size_t size);
		bool GetModel(std::vector<uint8_t>& outData);

		bool IsConnected() const { return redis != nullptr; }

	private:
		void* redis = nullptr;  // redisContext*
		std::string keyRollouts = "gigl:rollouts";
		std::string keyModel = "gigl:model";
	};
}

#endif

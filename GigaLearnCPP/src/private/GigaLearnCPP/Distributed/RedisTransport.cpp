#ifdef GIGL_REDIS

#include <GigaLearnCPP/Distributed/RedisTransport.h>
#include <hiredis.h>
#include <cstring>

namespace GGL {

RedisTransport::RedisTransport() = default;

RedisTransport::~RedisTransport() {
	Disconnect();
}

void RedisTransport::Disconnect() {
	if (redis) {
		redisFree((redisContext*)redis);
		redis = nullptr;
	}
}

bool RedisTransport::Connect(const std::string& host, int port) {
	Disconnect();
	redisContext* c = redisConnect(host.c_str(), port);
	if (!c || c->err) {
		if (c) redisFree(c);
		return false;
	}
	redis = c;
	return true;
}

bool RedisTransport::BlpopRollout(std::vector<uint8_t>& outData, int timeoutSec) {
	outData.clear();
	if (!redis) return false;

	redisReply* reply = (redisReply*)redisCommand((redisContext*)redis, "BLPOP %s %d", keyRollouts.c_str(), timeoutSec);
	if (!reply || reply->type != REDIS_REPLY_ARRAY || reply->elements < 2)
		return false;

	redisReply* dataReply = reply->element[1];
	if (dataReply->type != REDIS_REPLY_STRING) {
		freeReplyObject(reply);
		return false;
	}
	outData.assign((uint8_t*)dataReply->str, (uint8_t*)dataReply->str + dataReply->len);
	freeReplyObject(reply);
	return true;
}

bool RedisTransport::SetModel(const void* data, size_t size) {
	if (!redis) return false;
	redisReply* reply = (redisReply*)redisCommand((redisContext*)redis, "SET %b %b", keyModel.c_str(), keyModel.size(), data, size);
	bool ok = reply && reply->type != REDIS_REPLY_ERROR;
	if (reply) freeReplyObject(reply);
	return ok;
}

bool RedisTransport::RpushRollout(const void* data, size_t size) {
	if (!redis) return false;
	redisReply* reply = (redisReply*)redisCommand((redisContext*)redis, "RPUSH %b %b", keyRollouts.c_str(), keyRollouts.size(), data, size);
	bool ok = reply && reply->type != REDIS_REPLY_ERROR;
	if (reply) freeReplyObject(reply);
	return ok;
}

bool RedisTransport::GetModel(std::vector<uint8_t>& outData) {
	outData.clear();
	if (!redis) return false;
	redisReply* reply = (redisReply*)redisCommand((redisContext*)redis, "GET %s", keyModel.c_str());
	if (!reply || reply->type != REDIS_REPLY_STRING) {
		if (reply) freeReplyObject(reply);
		return false;
	}
	outData.assign((uint8_t*)reply->str, (uint8_t*)reply->str + reply->len);
	freeReplyObject(reply);
	return true;
}

}

#endif

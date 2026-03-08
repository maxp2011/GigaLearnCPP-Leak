#include "GigaLearnCPP/Distributed/DistributedTransport.h"
// Note: public header path

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef SOCKET socket_impl_t;
#define INVALID_SOCKET_IMPL INVALID_SOCKET
#define SOCKET_ERROR_IMPL SOCKET_ERROR
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
typedef int socket_impl_t;
#define INVALID_SOCKET_IMPL (-1)
#define SOCKET_ERROR_IMPL (-1)
#define closesocket close
#endif

#include <cstring>
#include <stdexcept>

namespace GGL {

static bool g_wsaInit = false;
static void EnsureWinsock() {
#ifdef _WIN32
	if (!g_wsaInit) {
		WSADATA wsa;
		if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0)
			g_wsaInit = true;
	}
#endif
}

void DistributedTransport::CloseSocket(socket_t& sock) {
	if (sock == INVALID_SOCKET) return;
	socket_impl_t s = (socket_impl_t)sock;
	closesocket(s);
	sock = INVALID_SOCKET;
}

bool DistributedTransport::SendAll(socket_t sock, const void* data, size_t size) {
	const uint8_t* p = (const uint8_t*)data;
	size_t sent = 0;
	while (sent < size) {
#ifdef _WIN32
		int n = send((SOCKET)sock, (const char*)(p + sent), (int)(size - sent), 0);
#else
		ssize_t n = send((int)sock, p + sent, size - sent, 0);
#endif
		if (n <= 0) return false;
		sent += (size_t)n;
	}
	return true;
}

bool DistributedTransport::RecvAll(socket_t sock, void* data, size_t size) {
	uint8_t* p = (uint8_t*)data;
	size_t got = 0;
	while (got < size) {
#ifdef _WIN32
		int n = recv((SOCKET)sock, (char*)(p + got), (int)(size - got), 0);
#else
		ssize_t n = recv((int)sock, p + got, size - got, 0);
#endif
		if (n <= 0) return false;
		got += (size_t)n;
	}
	return true;
}

DistributedTransport::~DistributedTransport() {
	StopServer();
	Disconnect();
	for (auto& s : clientSockets)
		CloseSocket(s);
	clientSockets.clear();
	CloseSocket(listenSocket);
}

bool DistributedTransport::StartServer(uint16_t port) {
	EnsureWinsock();
	CloseSocket(listenSocket);

	struct addrinfo hints = {}, *result = nullptr;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;
	hints.ai_flags = AI_PASSIVE;

	char portStr[16];
	snprintf(portStr, sizeof(portStr), "%u", port);

#ifdef _WIN32
	if (getaddrinfo(nullptr, portStr, &hints, &result) != 0)
		return false;
#else
	if (getaddrinfo(nullptr, portStr, &hints, &result) != 0)
		return false;
#endif

	socket_impl_t sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (sock == INVALID_SOCKET_IMPL) {
		freeaddrinfo(result);
		return false;
	}

	int opt = 1;
#ifdef _WIN32
	setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char*)&opt, sizeof(opt));
#else
	setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

	if (bind(sock, result->ai_addr, (int)result->ai_addrlen) == SOCKET_ERROR_IMPL) {
		closesocket(sock);
		freeaddrinfo(result);
		return false;
	}
	freeaddrinfo(result);

	if (listen(sock, SOMAXCONN) == SOCKET_ERROR_IMPL) {
		closesocket(sock);
		return false;
	}

	listenSocket = (socket_t)sock;
	return true;
}

void DistributedTransport::StopServer() {
	CloseSocket(listenSocket);
	for (auto& s : clientSockets)
		CloseSocket(s);
	clientSockets.clear();
}

int DistributedTransport::AcceptClient() {
	if (listenSocket == INVALID_SOCKET) return -1;

	socket_impl_t sock = accept((socket_impl_t)listenSocket, nullptr, nullptr);
	if (sock == INVALID_SOCKET_IMPL) return -1;

	int id = (int)clientSockets.size();
	clientSockets.push_back((socket_t)sock);
	return id;
}

int DistributedTransport::ReceiveFromClient(std::vector<uint8_t>& outData) {
	outData.clear();
	if (clientSockets.empty()) return -1;

	// Block on first client. (TODO: select/poll for multiple workers)
	int i = 0;
	uint32_t len = 0;
	if (!RecvAll(clientSockets[i], &len, 4)) {
		CloseSocket(clientSockets[i]);
		clientSockets.erase(clientSockets.begin() + i);
		return -1;
	}
	if (len > 100 * 1024 * 1024) return -1;  // Sanity: max 100MB
	outData.resize(len);
	if (!RecvAll(clientSockets[i], outData.data(), len)) {
		CloseSocket(clientSockets[i]);
		clientSockets.erase(clientSockets.begin() + i);
		return -1;
	}
	return i;
}

bool DistributedTransport::SendToClient(int clientId, const void* data, size_t size) {
	if (clientId < 0 || clientId >= (int)clientSockets.size()) return false;
	uint32_t len = (uint32_t)size;
	if (len != size) return false;
	if (!SendAll(clientSockets[clientId], &len, 4)) return false;
	if (!SendAll(clientSockets[clientId], data, size)) return false;
	return true;
}

bool DistributedTransport::Connect(const std::string& host, uint16_t port) {
	EnsureWinsock();
	Disconnect();

	struct addrinfo hints = {}, *result = nullptr;
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	char portStr[16];
	snprintf(portStr, sizeof(portStr), "%u", port);

	if (getaddrinfo(host.c_str(), portStr, &hints, &result) != 0)
		return false;

	socket_impl_t sock = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
	if (sock == INVALID_SOCKET_IMPL) {
		freeaddrinfo(result);
		return false;
	}

	if (connect(sock, result->ai_addr, (int)result->ai_addrlen) == SOCKET_ERROR_IMPL) {
		closesocket(sock);
		freeaddrinfo(result);
		return false;
	}
	freeaddrinfo(result);

	clientSocket = (socket_t)sock;
	return true;
}

void DistributedTransport::Disconnect() {
	CloseSocket(clientSocket);
}

bool DistributedTransport::Send(const void* data, size_t size) {
	if (clientSocket == INVALID_SOCKET) return false;
	uint32_t len = (uint32_t)size;
	if (len != size) return false;
	if (!SendAll(clientSocket, &len, 4)) return false;
	if (!SendAll(clientSocket, data, size)) return false;
	return true;
}

bool DistributedTransport::Receive(std::vector<uint8_t>& outData) {
	outData.clear();
	if (clientSocket == INVALID_SOCKET) return false;

	uint32_t len = 0;
	if (!RecvAll(clientSocket, &len, 4)) return false;
	if (len > 100 * 1024 * 1024) return false;
	outData.resize(len);
	if (!RecvAll(clientSocket, outData.data(), len)) return false;
	return true;
}

}

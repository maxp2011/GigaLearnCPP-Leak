#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <thread>

namespace GGL {

	// Simple TCP transport for distributed training.
	// Protocol: [4-byte length][payload]
	class DistributedTransport {
	public:
		~DistributedTransport();

		// --- Server (Learner) ---
		bool StartServer(uint16_t port);
		void StopServer();
		// Block until a client connects, returns client id (0-based)
		int AcceptClient();
		// Receive next message from any connected client. Returns client id, or -1 on error/closed.
		int ReceiveFromClient(std::vector<uint8_t>& outData);
		bool SendToClient(int clientId, const void* data, size_t size);

		// --- Client (Worker) ---
		bool Connect(const std::string& host, uint16_t port);
		void Disconnect();
		bool Send(const void* data, size_t size);
		bool Receive(std::vector<uint8_t>& outData);

		// --- Common ---
		int GetNumClients() const { return (int)clientSockets.size(); }
		bool IsConnected() const { return clientSocket != INVALID_SOCKET; }

	private:
		using socket_t = uintptr_t;  // SOCKET on Windows, int on Unix
		static constexpr socket_t INVALID_SOCKET = (socket_t)-1;

		socket_t listenSocket = INVALID_SOCKET;
		std::vector<socket_t> clientSockets;
		socket_t clientSocket = INVALID_SOCKET;  // For client mode

		bool SendAll(socket_t sock, const void* data, size_t size);
		bool RecvAll(socket_t sock, void* data, size_t size);
		void CloseSocket(socket_t& sock);
	};
}

#pragma once
#include "ActionParser.h"
#include <cmath>

namespace RLGC {

	// Pure continuous action parser - no hidden modifications
	// Network outputs 8 values in [-1, 1], directly mapped to game actions
	// Let the network learn everything through rewards
	//
	// Indices: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
	//
	class ContinuousAction : public ActionParser {
	public:

		static constexpr int ACTION_SIZE = 8;

		enum ActionIdx {
			THROTTLE = 0,
			STEER = 1,
			PITCH = 2,
			YAW = 3,
			ROLL = 4,
			JUMP = 5,
			BOOST = 6,
			HANDBRAKE = 7
		};

		// Use steer value for yaw when on ground (standard controller behavior)
		bool steerIsYaw;

		ContinuousAction(bool steerIsYaw = true) : steerIsYaw(steerIsYaw) {}

		// Main method - parse continuous action vector
		Action ParseContinuous(const float* actionValues, const Player& player, const GameState& state) {
			Action action = {};

			// Direct mapping - no deadzone, no modifications
			action.throttle = Clamp(actionValues[THROTTLE], -1.f, 1.f);
			action.steer = Clamp(actionValues[STEER], -1.f, 1.f);
			action.pitch = Clamp(actionValues[PITCH], -1.f, 1.f);
			action.roll = Clamp(actionValues[ROLL], -1.f, 1.f);

			// Yaw handling
			if (steerIsYaw && player.isOnGround) {
				action.yaw = action.steer;
			} else {
				action.yaw = Clamp(actionValues[YAW], -1.f, 1.f);
			}

			// Binary inputs - threshold at 0
			action.jump = actionValues[JUMP] > 0.f;
			action.boost = actionValues[BOOST] > 0.f;
			action.handbrake = actionValues[HANDBRAKE] > 0.f;

			return action;
		}

		// Parse from FList
		Action ParseContinuous(const FList& actionValues, const Player& player, const GameState& state) {
			return ParseContinuous(actionValues.data(), player, state);
		}

		// Required by interface but not used for continuous
		virtual Action ParseAction(int actionIdx, const Player& player, const GameState& state) override {
			return Action{};
		}

		virtual int GetActionAmount() override {
			return ACTION_SIZE;
		}

		virtual std::vector<uint8_t> GetActionMask(const Player& player, const GameState& state) override {
			return std::vector<uint8_t>(ACTION_SIZE, true);
		}

		// Get action space bounds (for PPO)
		static void GetActionBounds(float* low, float* high) {
			for (int i = 0; i < ACTION_SIZE; i++) {
				low[i] = -1.f;
				high[i] = 1.f;
			}
		}

	private:
		inline float Clamp(float value, float minVal, float maxVal) const {
			return value < minVal ? minVal : (value > maxVal ? maxVal : value);
		}
	};

} // namespace RLGC

#pragma once
#include <RLGymCPP/ObsBuilders/ObsBuilder.h>
#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Gamestates/Player.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <cmath>

namespace RLGC {

	// Comprehensive observation builder with useful game-sense features
	// Padding matches AdvancedObsPadderGGL structure for compatibility
	class StreamlinedObs : public ObsBuilder {
	public:
		int teamSize;
		float POS_STD;
		float ANG_STD;

		StreamlinedObs(int teamSize = 3);

		virtual FList BuildObs(const Player& player, const GameState& state) override;

	private:
		// Add zeros for missing player slot
		void AddDummy(FList& obs);

		// Add player data, returns PhysState for relative calculations
		PhysState AddPlayerToObs(FList& obs, const Player& player, const PhysState& ball, bool inv);
	};

} // namespace RLGC

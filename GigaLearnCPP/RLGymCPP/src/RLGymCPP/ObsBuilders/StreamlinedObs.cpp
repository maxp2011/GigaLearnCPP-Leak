#include "StreamlinedObs.h"
#include <RLGymCPP/Gamestates/StateUtil.h>
#include <vector>
#include <algorithm>

namespace RLGC {

StreamlinedObs::StreamlinedObs(int teamSize)
	: teamSize(teamSize), POS_STD(2300.0f), ANG_STD((float)M_PI) {
}

FList StreamlinedObs::BuildObs(const Player& player, const GameState& state) {
	FList obs;
	
	bool inverted = (player.team == Team::ORANGE);
	PhysState ball = InvertPhys(state.ball, inverted);
	const auto& pads = state.GetBoostPads(inverted);
	const auto& padTimers = state.GetBoostPadTimers(inverted);

	// ==========================================================
	// 1. BALL DATA (9 floats) - matches original
	// ==========================================================
	obs += ball.pos / POS_STD;
	obs += ball.vel / POS_STD;
	obs += ball.angVel / ANG_STD;

	// ==========================================================
	// 2. PREVIOUS ACTION (8 floats) - matches original
	// ==========================================================
	for (int i = 0; i < Action::ELEM_AMOUNT; i++) {
		obs += player.prevAction[i];
	}

	// ==========================================================
	// 3. BOOST PADS (34 floats) - with smooth respawn timers
	// ==========================================================
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		if (pads[i]) {
			obs += 1.f;
		} else {
			// Smooth respawn: big pads 10s, small pads 4s
			bool isBig = (i == 3 || i == 4 || i == 15 || i == 18 || i == 29 || i == 30);
			float maxTimer = isBig ? 10.f : 4.f;
			float availability = 1.f - (padTimers[i] / maxTimer);
			obs += std::max(0.f, availability);
		}
	}

	// ==========================================================
	// 4. THIS PLAYER'S DATA (26 floats) - matches original
	// ==========================================================
	PhysState playerCar = AddPlayerToObs(obs, player, ball, inverted);

	// ==========================================================
	// 5. EXTRA SELF FEATURES (useful additions)
	// ==========================================================
	
	// Speed and supersonic (2 floats)
	float speed = playerCar.vel.Length();
	obs += speed / POS_STD;
	obs += (speed > 2200.f) ? 1.f : 0.f;

	// Local velocity - movement relative to facing (3 floats)
	Vec localVel = playerCar.rotMat.Dot(playerCar.vel);
	obs += localVel / POS_STD;

	// Wall/ceiling proximity (4 floats) - for wall plays and recoveries
	float wallDistX = std::min(
		std::abs(playerCar.pos.x - CommonValues::SIDE_WALL_X),
		std::abs(playerCar.pos.x + CommonValues::SIDE_WALL_X)
	);
	float wallDistY = std::min(
		std::abs(playerCar.pos.y - CommonValues::BACK_WALL_Y),
		std::abs(playerCar.pos.y + CommonValues::BACK_WALL_Y)
	);
	float ceilingDist = CommonValues::CEILING_Z - playerCar.pos.z;
	float floorDist = playerCar.pos.z - 17.f;

	obs += std::max(0.f, 1.f - wallDistX / 500.f);
	obs += std::max(0.f, 1.f - wallDistY / 500.f);
	obs += std::max(0.f, 1.f - ceilingDist / 500.f);
	obs += std::max(0.f, 1.f - floorDist / 200.f);

	// Ball height and aerial indicator (2 floats)
	obs += ball.pos.z / POS_STD;
	obs += (ball.pos.z > 300.f) ? 1.f : ball.pos.z / 300.f;

	// Facing ball and closing speed (2 floats)
	Vec toBall = (ball.pos - playerCar.pos).Normalized();
	float facingBall = playerCar.rotMat.forward.Dot(toBall);
	float closingSpeed = playerCar.vel.Dot(toBall);
	obs += facingBall;
	obs += closingSpeed / POS_STD;

	// ==========================================================
	// 6. GOAL AWARENESS (12 floats)
	// ==========================================================
	Vec ownGoal = Vec(0, -CommonValues::BACK_WALL_Y + 100.f, CommonValues::GOAL_HEIGHT / 2);
	Vec enemyGoal = Vec(0, CommonValues::BACK_WALL_Y - 100.f, CommonValues::GOAL_HEIGHT / 2);

	// Goal positions relative to self
	obs += (ownGoal - playerCar.pos) / POS_STD;
	obs += (enemyGoal - playerCar.pos) / POS_STD;

	// Ball to goal directions (for shot/defensive reads)
	Vec ballToEnemy = (enemyGoal - ball.pos).Normalized();
	Vec ballToOwn = (ownGoal - ball.pos).Normalized();
	obs += ballToEnemy;
	obs += ballToOwn;

	// ==========================================================
	// 7. ALLIES with padding (teamSize-1 slots × 32 floats each)
	// ==========================================================
	std::vector<const Player*> allies;
	std::vector<const Player*> enemies;
	
	for (const auto& other : state.players) {
		if (other.carId == player.carId) continue;
		(other.team == player.team ? allies : enemies).push_back(&other);
	}

	// Shuffle to prevent slot bias
	std::shuffle(allies.begin(), allies.end(), ::Math::GetRandEngine());
	std::shuffle(enemies.begin(), enemies.end(), ::Math::GetRandEngine());

	int allyCount = 0;
	for (const auto* ally : allies) {
		if (allyCount >= teamSize - 1) break;
		PhysState otherCar = AddPlayerToObs(obs, *ally, ball, inverted);
		obs += (otherCar.pos - playerCar.pos) / POS_STD;
		obs += (otherCar.vel - playerCar.vel) / POS_STD;
		allyCount++;
	}
	while (allyCount < teamSize - 1) {
		AddDummy(obs);
		allyCount++;
	}

	// ==========================================================
	// 8. ENEMIES with padding (teamSize slots × 32 floats each)
	// ==========================================================
	int enemyCount = 0;
	for (const auto* enemy : enemies) {
		if (enemyCount >= teamSize) break;
		PhysState otherCar = AddPlayerToObs(obs, *enemy, ball, inverted);
		obs += (otherCar.pos - playerCar.pos) / POS_STD;
		obs += (otherCar.vel - playerCar.vel) / POS_STD;
		enemyCount++;
	}
	while (enemyCount < teamSize) {
		AddDummy(obs);
		enemyCount++;
	}

	// ==========================================================
	// 9. TEAM AGGREGATE INFO (6 floats) - quick situational reads
	// ==========================================================
	float closestOppToBall = 99999.f;
	float closestTeammateToBall = 99999.f;
	float totalTeamBoost = player.boost;
	float totalOppBoost = 0.f;
	int aliveTeammates = 1;
	int aliveOpponents = 0;

	for (const auto& other : state.players) {
		if (other.carId == player.carId) continue;

		PhysState otherPhys = InvertPhys(other, inverted);
		float distToBall = (otherPhys.pos - ball.pos).Length();

		if (other.team == player.team) {
			closestTeammateToBall = std::min(closestTeammateToBall, distToBall);
			totalTeamBoost += other.boost;
			if (!other.isDemoed) aliveTeammates++;
		} else {
			closestOppToBall = std::min(closestOppToBall, distToBall);
			totalOppBoost += other.boost;
			if (!other.isDemoed) aliveOpponents++;
		}
	}

	float myDistToBall = (playerCar.pos - ball.pos).Length();

	// Am I closest to ball on my team?
	obs += (myDistToBall <= closestTeammateToBall) ? 1.f : 0.f;

	// Distance advantage over closest opponent
	float distAdvantage = (closestOppToBall - myDistToBall) / POS_STD;
	obs += std::tanh(distAdvantage);

	// Team boost advantage
	float boostAdvantage = (totalTeamBoost - totalOppBoost) / 100.f;
	obs += std::tanh(boostAdvantage / (float)teamSize);

	// Alive counts
	obs += aliveTeammates / (float)teamSize;
	obs += aliveOpponents / (float)teamSize;

	// Behind ball (defensive position)
	Vec selfToBall = (ball.pos - playerCar.pos).Normalized();
	float behindBall = selfToBall.Dot(ballToOwn);
	obs += behindBall;

	return obs;
}

void StreamlinedObs::AddDummy(FList& obs) {
	// 26 floats for base player data
	for (int i = 0; i < 7; i++) obs += {0, 0, 0};
	obs += {0, 0, 0, 0, 0};

	// 6 floats for relative pos/vel
	obs += {0, 0, 0};
	obs += {0, 0, 0};
}

PhysState StreamlinedObs::AddPlayerToObs(FList& obs, const Player& player, const PhysState& ball, bool inv) {
	PhysState playerCar = InvertPhys(player, inv);

	Vec relPos = ball.pos - playerCar.pos;
	Vec relVel = ball.vel - playerCar.vel;

	// Player physics data (21 floats)
	obs += relPos / POS_STD;
	obs += relVel / POS_STD;
	obs += playerCar.pos / POS_STD;
	obs += playerCar.rotMat.forward;
	obs += playerCar.rotMat.up;
	obs += playerCar.vel / POS_STD;
	obs += playerCar.angVel / ANG_STD;

	// Player state data (5 floats)
	obs += FList{
		player.boost / 100.f,
		(float)player.isOnGround,
		(float)player.HasFlipOrJump(),
		(float)player.isDemoed,
		(float)player.hasJumped  // Useful for flip reset detection instead of placeholder
	};

	return playerCar;
}

} // namespace RLGC

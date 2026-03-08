#pragma once

#include "AdvancedObs.h"
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Gamestates/StateUtil.h>
#include "../../../RocketSim/src/RLConst.h"
#include <cmath>

namespace RLGC {

	// Custom observation builder — non-standard feature layout, derived physics features,
	// dash timers, and unique normalizations. Do not distribute.
	class CustomObs : public AdvancedObs {
	public:
		static constexpr int MAX_TEAMMATES  = 2;
		static constexpr int MAX_OPPONENTS  = 3;

		// Base AdvancedObs per-player = 29, plus 9 custom features + 1 team flag + 5 dash timers = 44
		static constexpr int PLAYER_FEAT_SIZE = 44;

		// Dash timer normalization (match RLConst)
		static constexpr float JUMP_TIME_MAX   = 0.2f;   // JUMP_MAX_TIME
		static constexpr float FLIP_TIME_MAX   = 0.65f;   // FLIP_TORQUE_TIME
		static constexpr float AIR_TIME_MAX    = 1.25f;   // DOUBLEJUMP_MAX_DELAY
		static constexpr float BOOST_TIME_MAX  = 0.1f;    // BOOST_MIN_TIME

		// --- Custom normalization constants (non-standard, unique to this obs) ---
		static constexpr float DIST_COEF       = 1.f / 3000.f;   // field ~4096 x 5120
		static constexpr float SPEED_COEF      = 1.f / 2300.f;   // max car speed
		static constexpr float BALL_SPEED_COEF = 1.f / 6000.f;   // max ball speed (high aerial shots)
		static constexpr float TIME_COEF       = 1.f / 5.f;      // time estimates in seconds

		// Goal positions (blue scores into orange goal at +y)
		static constexpr float GOAL_Y    = 5120.f;
		static constexpr float GOAL_Z    =  321.f;

		// -----------------------------------------------------------------------
		// Helper: goal positions from perspective of player team
		// -----------------------------------------------------------------------
		static Vec OwnGoalPos(bool inv) {
			// If blue (not inv), own goal is -y. If orange (inv), own goal is +y.
			return Vec(0.f, inv ? GOAL_Y : -GOAL_Y, GOAL_Z);
		}
		static Vec EnemyGoalPos(bool inv) {
			return Vec(0.f, inv ? -GOAL_Y : GOAL_Y, GOAL_Z);
		}

		// -----------------------------------------------------------------------
		// Helper: simple time-to-ball heuristic (car dist / (car speed + epsilon))
		// -----------------------------------------------------------------------
		static float TimeToBall(const PhysState& car, const PhysState& ball) {
			float dist = (ball.pos - car.pos).Length();
			float spd  = car.vel.Length();
			return dist / (spd + 500.f); // +500 avoids div-by-zero and biases toward aggression
		}

		// -----------------------------------------------------------------------
		// Helper: signed progress of ball toward enemy goal (dot of ball vel with goal dir)
		// -----------------------------------------------------------------------
		static float BallGoalProgress(const PhysState& ball, bool inv) {
			// +y is enemy goal for blue. ball vel y component normalised.
			float dir = inv ? -1.f : 1.f;
			return (ball.vel.y * dir) * BALL_SPEED_COEF;
		}

		// -----------------------------------------------------------------------
		// Adds 29 (AdvancedObs base) + 9 (custom) + 5 (dash timers) + 1 (team flag) = 44 features per player
		// -----------------------------------------------------------------------
		void AddCustomPlayerToObs(
			FList& out,
			const Player& player,
			bool inv,
			const PhysState& ball,
			bool isClosestToBall,
			bool isClosestOnTeam,
			const Vec& ownGoal,
			const Vec& enemyGoal,
			float teamFlag  // 1.0 = teammate, -1.0 = opponent, 0.0 = self
		) {
			// --- 29 standard features from parent ---
			AddPlayerToObs(out, player, inv, ball);

			// --- 9 custom derived features ---

			// 1. Normalised distance to ball
			float distBall = (ball.pos - player.pos).Length();
			out += distBall * DIST_COEF;

			// 2. Normalised car speed (scalar — not included in base)
			float carSpeed = player.vel.Length();
			out += carSpeed * SPEED_COEF;

			// 3. Normalised distance to own goal
			float distOwnGoal = (ownGoal - player.pos).Length();
			out += distOwnGoal * DIST_COEF;

			// 4. Normalised distance to enemy goal
			float distEnemyGoal = (enemyGoal - player.pos).Length();
			out += distEnemyGoal * DIST_COEF;

			// 5. Time-to-ball estimate (clamped 0-1)
			float ttb = TimeToBall(player, ball);
			out += std::min(ttb * TIME_COEF, 1.f);

			// 6. Is this player the globally closest car to ball?
			out += isClosestToBall ? 1.f : 0.f;

			// 7. Is this player the closest on their team to ball?
			out += isClosestOnTeam ? 1.f : 0.f;

			// 8. Alignment: how much is the car pointing toward ball? (-1..1)
			Vec toBall = (ball.pos - player.pos);
			float toBallLen = toBall.Length();
			float alignment = 0.f;
			if (toBallLen > 1.f) {
				Vec toBallNorm = toBall * (1.f / toBallLen);
				// Forward vector of car (first column of rotation matrix)
				alignment = player.rotMat.forward.Dot(toBallNorm);
			}
			out += alignment;

			// 9. Ball vel component toward enemy goal (ball "danger" from this player's POV)
			out += BallGoalProgress(ball, inv);

			// 10-14. Dash timers (normalised 0-1) — critical for flip resets, wavedashes, double jumps
			out += std::min(player.jumpTime / JUMP_TIME_MAX, 1.f);
			out += std::min(player.flipTime / FLIP_TIME_MAX, 1.f);
			out += std::min(player.airTimeSinceJump / AIR_TIME_MAX, 1.f);
			out += std::min(player.timeSpentBoosting / BOOST_TIME_MAX, 1.f);
			out += std::min(std::max(player.handbrakeVal, 0.f), 1.f);

			// 15. Team relationship flag: 1.0 = teammate, -1.0 = opponent, 0.0 = self
			// Lets the network distinguish friend from foe in a single clear signal
			out += teamFlag;
		}

		// -----------------------------------------------------------------------
		// Main obs builder
		// -----------------------------------------------------------------------
		virtual FList BuildObs(const Player& player, const GameState& state) override {
			FList obs = {};

			bool inv = (player.team == Team::ORANGE);

			auto ball    = InvertPhys(state.ball, inv);
			auto& pads      = state.GetBoostPads(inv);
			auto& padTimers = state.GetBoostPadTimers(inv);

			Vec ownGoal   = OwnGoalPos(inv);
			Vec enemyGoal = EnemyGoalPos(inv);

			// ----------------------------------------------------------------
			// --- BALL SECTION ---
			// ----------------------------------------------------------------
			obs += ball.pos * POS_COEF;
			obs += ball.vel  / CommonValues::BALL_MAX_SPEED;
			obs += ball.angVel * ANG_VEL_COEF;

			// Custom: ball speed scalar
			obs += ball.vel.Length() * BALL_SPEED_COEF;

			// Custom: normalised ball-to-own-goal and ball-to-enemy-goal vectors
			Vec ballToOwn   = (ownGoal   - ball.pos);
			Vec ballToEnemy = (enemyGoal - ball.pos);
			obs += ballToOwn   * DIST_COEF;   // 3 floats
			obs += ballToEnemy * DIST_COEF;   // 3 floats

			// Custom: ball progress toward enemy goal
			obs += BallGoalProgress(ball, inv);

			// ----------------------------------------------------------------
			// --- PREVIOUS ACTION ---
			// ----------------------------------------------------------------
			for (int i = 0; i < player.prevAction.ELEM_AMOUNT; i++)
				obs += player.prevAction[i];

			// ----------------------------------------------------------------
			// --- BOOST PADS (with smooth timer decay) ---
			// ----------------------------------------------------------------
			for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
				if (pads[i]) {
					obs += 1.f;
				} else {
					// Smooth exponential decay instead of 1/(1+t)
					obs += std::exp(-padTimers[i] * 0.2f);
				}
			}

			// ----------------------------------------------------------------
			// --- Precompute: who is closest to ball globally / per-team ---
			// ----------------------------------------------------------------
			int   closestGlobalId  = -1;
			int   closestBlueId    = -1;
			int   closestOrangeId  = -1;
			float closestGlobalDist = 1e9f;
			float closestBlueDist   = 1e9f;
			float closestOrangeDist = 1e9f;

			for (auto& p : state.players) {
				if (p.isDemoed) continue;
				float d = (state.ball.pos - p.pos).Length();
				if (d < closestGlobalDist) {
					closestGlobalDist = d;
					closestGlobalId   = p.carId;
				}
				if (p.team == Team::BLUE && d < closestBlueDist) {
					closestBlueDist = d;
					closestBlueId   = p.carId;
				}
				if (p.team == Team::ORANGE && d < closestOrangeDist) {
					closestOrangeDist = d;
					closestOrangeId   = p.carId;
				}
			}

			// ----------------------------------------------------------------
			// --- SELF ---
			// ----------------------------------------------------------------
			bool selfClosestGlobal = (player.carId == closestGlobalId);
			bool selfClosestTeam   = (player.carId == (inv ? closestOrangeId : closestBlueId));
			AddCustomPlayerToObs(obs, player, inv, ball, selfClosestGlobal, selfClosestTeam, ownGoal, enemyGoal, 0.f);

			// ----------------------------------------------------------------
			// --- TEAMMATES + OPPONENTS ---
			// ----------------------------------------------------------------
			FList teammates = {}, opponents = {};

			for (auto& other : state.players) {
				if (other.carId == player.carId) continue;

				bool isTeammate = (other.team == player.team);
				auto otherPhys  = InvertPhys(other, inv);

				bool otherClosestGlobal = (other.carId == closestGlobalId);
				bool otherClosestTeam;
				if (isTeammate)
					otherClosestTeam = (other.carId == (inv ? closestOrangeId : closestBlueId));
				else
					otherClosestTeam = (other.carId == (inv ? closestBlueId   : closestOrangeId));

				FList& target = isTeammate ? teammates : opponents;
				float teamFlag = isTeammate ? 1.f : -1.f;
				AddCustomPlayerToObs(target, other, inv, ball, otherClosestGlobal, otherClosestTeam, ownGoal, enemyGoal, teamFlag);
			}

			// ----------------------------------------------------------------
			// --- PAD + CLAMP teammates/opponents to MAX ---
			// ----------------------------------------------------------------
			auto padZeros = [&](FList& out, int missing) {
				int elems = missing * PLAYER_FEAT_SIZE;
				for (int i = 0; i < elems; i++) out += 0.f;
			};

			int curTM = (int)(teammates.size() / PLAYER_FEAT_SIZE);
			int missTM = std::max(0, MAX_TEAMMATES - curTM);
			obs += teammates;
			padZeros(obs, missTM);

			int curOpp = (int)(opponents.size() / PLAYER_FEAT_SIZE);
			int missOpp = std::max(0, MAX_OPPONENTS - curOpp);
			obs += opponents;
			padZeros(obs, missOpp);

			return obs;
		}
	};
}

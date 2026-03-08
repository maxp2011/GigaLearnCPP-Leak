#pragma once
#include "Reward.h"
#include "../Math.h"

namespace RLGC {

	using namespace CommonValues;

	inline float distance2D(float x1, float z1, float x2, float z2) {
		float dx = x2 - x1;
		float dz = z2 - z1;
		return std::sqrt(dx * dx + dz * dz);
	}

	inline std::vector<float> linspace(float start_in, float end_in, int num_in) {
		std::vector<float> linspaced;

		if (num_in <= 0) {
			return linspaced;
		}
		if (num_in == 1) {
			linspaced.push_back(start_in);
			return linspaced;
		}

		float delta = (end_in - start_in) / (num_in - 1);
		for (int i = 0; i < num_in; ++i) {
			linspaced.push_back(start_in + delta * i);
		}
		return linspaced;
	}

	template<bool PlayerEventState::* VAR, bool NEGATIVE>
	class PlayerDataEventReward : public Reward {
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool val = player.eventState.*VAR;

			if (NEGATIVE) {
				return -(float)val;
			}
			else {
				return (float)val;
			}
		}
	};



	typedef PlayerDataEventReward<&PlayerEventState::goal, false> PlayerGoalReward; // NOTE: Given only to the player who last touched the ball on the opposing team
	typedef PlayerDataEventReward<&PlayerEventState::assist, false> AssistReward;
	typedef PlayerDataEventReward<&PlayerEventState::shot, false> ShotReward;
	typedef PlayerDataEventReward<&PlayerEventState::shotPass, false> ShotPassReward;
	typedef PlayerDataEventReward<&PlayerEventState::save, false> SaveReward;
	typedef PlayerDataEventReward<&PlayerEventState::bump, false> BumpReward;
	typedef PlayerDataEventReward<&PlayerEventState::bumped, true> BumpedPenalty;
	typedef PlayerDataEventReward<&PlayerEventState::demo, false> DemoReward;
	typedef PlayerDataEventReward<&PlayerEventState::demoed, true> DemoedPenalty;

	// Rewards a goal by anyone on the team
	// NOTE: Already zero-sum
	class GoalReward : public Reward {
	public:
		float concedeScale;
		GoalReward(float concedeScale = -.5) : concedeScale(concedeScale) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!state.goalScored)
				return 0;

			bool scored = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));
			return scored ? 1 : concedeScale;
		}
	};



class RippleDemoReward : public Reward {
public:
    float demo_attacker;
    float demo_victim;

    RippleDemoReward(float demo_attacker = 5.f, float demo_victim = 1.f)
        : demo_attacker(demo_attacker), demo_victim(demo_victim) {}

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        if (!state.prev) return 0.f;

        const Player* prev = nullptr;
        for (const auto& p : state.prev->players) {
            if (p.carId == player.carId) {
                prev = &p;
                break;
            }
        }
        if (!prev) return 0.f;

        float reward = 0.f;

        if (player.isDemoed && !prev->isDemoed) {
            reward -= demo_victim;
        }

        if (player.eventState.demo && !prev->eventState.demo) {
            reward += demo_attacker;
        }

        return reward;
    }
};
	


// Simple dense reward to teach the bot how to gain flip resets.
//
// Three signals, always active in the air:
//
//   1. Proximity to ball   - get close to the ball
//   2. Wheel alignment     - point the underside of the car at the ball
//      (-rotMat.up . dirToBall), gated so it only rewards when also close
//   3. Reset bonus         - one-time spike when HasFlipReset() is gained
//
// Intentionally minimal. No staging, no chaining, no conditions.
// Just nudges the bot toward the ball upside-down until it stumbles
// into a reset, then rewards it clearly for doing so.



    class PopResetReward : public Reward {
public:
    explicit PopResetReward(
        float scale = 1.0f,
        float flipAfterResetWindowSec = 0.45f,
        float maxArmedWaitSec = 4.0f,
        int   maxTouchAttempts = 7,
        float minUpwardVelZGain = 30.0f,
        float minDeltaVNorm     = 150.0f,
        float deltaVelScale = 1.0f,
        float zSqWeight     = 2.0f,
        bool  onlyUpwardZ   = true,
        float popSaturation = 200000.0f,

        float mult1     = 1.0f,
        float mult2     = 8.0f,
        float mult3     = 16.0f,
        float mult4     = 2.0f,
        float mult5Plus = 2.0f,

        float groundMaxCarZ      = 170.0f,
        float wallNearDist       = 250.0f,
        float wallMaxCarZ        = 380.0f,
        float maxAbsVzForGround  = 350.0f,
        float groundGraceSec     = 0.03f,

        float minAirZForReset = 220.0f,

        float angleStartDeg   = 35.0f,
        float angleEndDeg     = 60.0f,
        float angleMinFactor  = 0.1f,
        bool  angleSmoothStep = true,

        float xyKmhMin   = 10.0f,
        float xyKmhMax   = 60.0f,
        float xyMultMin  = 0.2f,
        float xyMultMax  = 1.0f,

        float heightMultAtLow   = 0.2f,
        float heightMultAtHigh  = 1.0f,
        float heightLowFracGoal = 2.0f / 3.0f,
        float heightHighFracCeil= 0.55f,
        float disableHeightNearGoalXYDist = 1400.0f,

        // wasted pop penalty
        float wastedPopPenalty = 0.0f,
        float wastedPopMaxWatchSec = 0.0f,
        float wastedPopGroundDiamMult = 0.0f,
        float disableWastedPopNearGoalXYDist = 2500.0f,
        float ballWallTouchDist = 120.0f,
        float ballWallTouchHyst = 30.0f,

        // ===== Field-position multiplier =====
        // signedY := (BLUE ? ball.y : -ball.y) so enemy side is +
        // own backboard (-BACK_WALL_Y) => ownBackboardMult
        // center (0) => midlineMult
        // enemy-half midline (+BACK_WALL_Y/2) => enemyHalfMidlineMult
        // if ball is within (enemyGoalCloseBallDiam * ballDiameter) of enemy goal center in XY => force enemyGoalCloseMult
        float ownBackboardMult = 0.3f,
        float midlineMult = 0.7f,
        float enemyHalfMidlineMult = 1.0f,
        float enemyGoalCloseMult = 0.1f,
        float enemyGoalCloseBallDiam = 3.5f,

        // ===== No-bounce goal bonus after pop-success =====
        float noBounceGoalBonus = 2.0f,
        float noBounceMaxWatchSec = 6.0f,
        float noBounceGroundDiamMult = 1.8f,
        float noBounceWallTouchDist = 120.0f,
        float noBounceWallTouchHyst = 30.0f,
        // NEW: only grant no-bounce bonus if pop-count in the chain is >= this
        int   noBounceMinPopCount = 2,

        // ===== Own-half goal-side 2/5: penalize success instead of counting =====
        // own half is signedY in [-BACK_WALL_Y, 0]
        // goal-side 2/5 => signedY in [-BACK_WALL_Y, -(3/5)BACK_WALL_Y]
        bool  penalizeSuccessInOwnGoalSideFrac = true,
        float ownGoalSideFracOfHalf           = 2.0f / 5.0f,
        float ownGoalSideSuccessPenalty       = 8.35f
    )
    : scale(scale),
      flipAfterResetWindowSec(std::max(0.01f, flipAfterResetWindowSec)),
      maxArmedWaitSec(std::max(0.05f, maxArmedWaitSec)),
      maxTouchAttempts(std::max(1, maxTouchAttempts)),
      minUpwardVelZGain(minUpwardVelZGain),
      minDeltaVNorm(std::max(0.0f, minDeltaVNorm)),
      deltaVelScale(deltaVelScale),
      zSqWeight(zSqWeight),
      onlyUpwardZ(onlyUpwardZ),
      popSaturation(std::max(1.0f, popSaturation)),
      mult1(mult1), mult2(mult2), mult3(mult3), mult4(mult4), mult5Plus(mult5Plus),
      groundMaxCarZ(groundMaxCarZ),
      wallNearDist(wallNearDist),
      wallMaxCarZ(std::max(groundMaxCarZ, wallMaxCarZ)),
      maxAbsVzForGround(std::max(0.0f, maxAbsVzForGround)),
      groundGraceSec(std::max(0.0f, groundGraceSec)),
      minAirZForReset(std::max(0.0f, minAirZForReset)),
      angleStartDeg(std::max(0.0f, angleStartDeg)),
      angleEndDeg(std::max(angleStartDeg + 0.1f, angleEndDeg)),
      angleMinFactor(std::clamp(angleMinFactor, 0.0f, 1.0f)),
      angleSmoothStep(angleSmoothStep),
      xyKmhMin(std::max(0.0f, xyKmhMin)),
      xyKmhMax(std::max(xyKmhMin + 0.1f, xyKmhMax)),
      xyMultMin(std::clamp(xyMultMin, 0.0f, 10.0f)),
      xyMultMax(std::clamp(xyMultMax, 0.0f, 10.0f)),
      heightMultAtLow(std::max(0.0f, heightMultAtLow)),
      heightMultAtHigh(std::max(0.0f, heightMultAtHigh)),
      heightLowFracGoal(std::max(0.0f, heightLowFracGoal)),
      heightHighFracCeil(std::max(0.0f, heightHighFracCeil)),
      disableHeightNearGoalXYDist(std::max(0.0f, disableHeightNearGoalXYDist)),
      wastedPopPenalty(std::max(0.0f, wastedPopPenalty)),
      wastedPopMaxWatchSec(std::max(0.1f, wastedPopMaxWatchSec)),
      wastedPopGroundDiamMult(std::max(0.1f, wastedPopGroundDiamMult)),
      disableWastedPopNearGoalXYDist(std::max(0.0f, disableWastedPopNearGoalXYDist)),
      ballWallTouchDist(std::max(0.0f, ballWallTouchDist)),
      ballWallTouchHyst(std::max(0.0f, ballWallTouchHyst)),
      ownBackboardMult(std::max(0.0f, ownBackboardMult)),
      midlineMult(std::max(0.0f, midlineMult)),
      enemyHalfMidlineMult(std::max(0.0f, enemyHalfMidlineMult)),
      enemyGoalCloseMult(std::max(0.0f, enemyGoalCloseMult)),
      enemyGoalCloseBallDiam(std::max(0.0f, enemyGoalCloseBallDiam)),
      noBounceGoalBonus(std::max(0.0f, noBounceGoalBonus)),
      noBounceMaxWatchSec(std::max(0.05f, noBounceMaxWatchSec)),
      noBounceGroundDiamMult(std::max(0.1f, noBounceGroundDiamMult)),
      noBounceWallTouchDist(std::max(0.0f, noBounceWallTouchDist)),
      noBounceWallTouchHyst(std::max(0.0f, noBounceWallTouchHyst)),
      noBounceMinPopCount(std::max(1, noBounceMinPopCount)),
      penalizeSuccessInOwnGoalSideFrac(penalizeSuccessInOwnGoalSideFrac),
      ownGoalSideFracOfHalf(std::clamp(ownGoalSideFracOfHalf, 0.0f, 0.999f)),
      ownGoalSideSuccessPenalty(std::max(0.0f, ownGoalSideSuccessPenalty)) {}

    void Reset(const GameState&) override {
        states.clear();
    }

    float GetReward(const Player& player, const GameState& state, bool) override {
        auto& s = states[player.carId];
        const float dt = (state.deltaTime > 0.f) ? state.deltaTime : (1.f / 120.f);

        // goal edge (goalScored is global)
        const bool goalNow = state.goalScored;
        s.goalEdge = (!s.prevGoalScored && goalNow);
        s.prevGoalScored = goalNow;

        float rewardRaw = 0.f;

        // Resolve no-bounce bonus watcher (goals occurring after pop)
        rewardRaw += UpdateNoBounceGoalBonus(player, state, dt, s);

        const bool realGround = IsRealGround(player, dt, s.groundGrace);

        if (realGround) {
            s.successCount = 0;

            s.pending = false;
            s.armed = false;
            s.pendingTimer = 0.f;
            s.armedTimer = 0.f;
            s.touchAttempts = 0;
            s.sawFlipAvail = false;
            s.flipUsed = false;

            s.wasteWatch = false;
            s.wasteTimer = 0.f;
        }

        rewardRaw += UpdateWastedPopPenalty(player, state, dt, s);

        const bool hasFlipNow  = player.HasFlipOrJump();
        const bool gotResetNow = player.GotFlipReset();

        const bool gotResetEdge = (!s.prevGotReset && gotResetNow);
        s.prevGotReset = gotResetNow;

        const bool flipRegainedEdge = (!s.prevHasFlip && hasFlipNow);
        s.prevHasFlip = hasFlipNow;

        const bool airborneEnough = (!realGround && player.pos.z >= minAirZForReset);
        const bool resetAcquired = airborneEnough && (flipRegainedEdge || gotResetEdge);

        if (resetAcquired) {
            s.pending = true;
            s.pendingTimer = 0.f;

            s.sawFlipAvail = hasFlipNow;
            s.flipUsed = false;

            s.armed = false;
            s.armedTimer = 0.f;
            s.touchAttempts = 0;

            if (gotResetEdge && !hasFlipNow) {
                s.flipUsed = true;
            }
        }

        if (s.pending) {
            s.pendingTimer += dt;

            if (hasFlipNow) s.sawFlipAvail = true;
            if (s.sawFlipAvail && !hasFlipNow) s.flipUsed = true;

            if (s.flipUsed && s.pendingTimer <= flipAfterResetWindowSec) {
                s.armed = true;
                s.pending = false;
                s.armedTimer = 0.f;
                s.touchAttempts = 0;
            }

            if (s.pendingTimer > flipAfterResetWindowSec) {
                s.pending = false;
            }
        }

        if (s.armed) {
            s.armedTimer += dt;

            if (s.armedTimer > maxArmedWaitSec) {
                s.armed = false;
                s.armedTimer = 0.f;
                s.touchAttempts = 0;
            }
            else if (player.ballTouchedStep && state.prev) {
                s.touchAttempts += 1;

                const Vec& vb = state.prev->ball.vel;
                const Vec& va = state.ball.vel;

                const float dvx = va.x - vb.x;
                const float dvy = va.y - vb.y;
                const float dvz = va.z - vb.z;

                const float dvNorm = std::sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
                const bool success = (dvz >= minUpwardVelZGain) && (dvNorm >= minDeltaVNorm);

                if (success) {
                    // In own-half goal-side 2/5: penalize and do NOT count as success
                    if (penalizeSuccessInOwnGoalSideFrac && IsInOwnGoalSideFracOfHalf(player, state)) {
                        rewardRaw -= ownGoalSideSuccessPenalty;

                        // consume attempt, but do NOT: reward, chain increment, watchers
                        s.armed = false;
                        s.armedTimer = 0.f;
                        s.touchAttempts = 0;
                        // IMPORTANT: successCount unchanged
                    } else {
                        float dzUse = dvz;
                        if (onlyUpwardZ) dzUse = std::max(dzUse, 0.f);

                        const float popScore = dvx*dvx + dvy*dvy + zSqWeight * dzUse*dzUse;
                        const float popRatio = popScore / (popScore + popSaturation);

                        const float chainMult = GetChainMult(s.successCount);
                        float successReward = (popRatio * deltaVelScale) * chainMult;

                        successReward *= GetGoalAngleFactor(player, state);
                        successReward *= GetXySpeedFactor(state);
                        successReward *= GetHeightFactor(player, state);
                        successReward *= GetFieldPosMult(player, state);

                        rewardRaw += successReward;

                        StartWastedPopWatch(player, state, s);

                        // No-bounce watcher ONLY if this is pop #>= noBounceMinPopCount
                        const int newCount = s.successCount + 1; // include this success
                        if (newCount >= noBounceMinPopCount) {
                            StartNoBounceGoalWatch(state, s);
                        } else {
                            // ensure no stale watcher remains
                            s.nbWatch = false;
                            s.nbValid = false;
                            s.nbTimer = 0.f;
                        }

                        // If a goal happens on the same tick as success, pay immediately too (ONLY if allowed)
                        if (s.goalEdge && state.goalScored) {
                            const bool allowNoBounceNow = (newCount >= noBounceMinPopCount);
                            if (allowNoBounceNow && s.nbValid && IsOppGoalScored(player, state)) {
                                rewardRaw += noBounceGoalBonus;
                            }
                            s.nbWatch = false;
                            s.nbValid = false;
                            s.nbTimer = 0.f;
                        }

                        s.successCount += 1;

                        s.armed = false;
                        s.armedTimer = 0.f;
                        s.touchAttempts = 0;
                    }
                }
                else {
                    if (s.touchAttempts >= maxTouchAttempts) {
                        s.armed = false;
                        s.armedTimer = 0.f;
                        s.touchAttempts = 0;
                    }
                }
            }
        }

        float out = rewardRaw * scale;
        if (!std::isfinite(out)) {
            const float maxOut = 1.0e9f;
            out = (out >= 0.f) ? maxOut : -maxOut;
        }
        return out;
    }

private:
    struct PerCarState {
        int  successCount = 0;

        bool prevHasFlip  = false;
        bool prevGotReset = false;

        bool  pending = false;
        float pendingTimer = 0.f;

        bool  sawFlipAvail = false;
        bool  flipUsed = false;

        bool  armed = false;
        float armedTimer = 0.f;

        int   touchAttempts = 0;

        float groundGrace = 0.f;

        // wasted pop watch
        bool  wasteWatch = false;
        float wasteTimer = 0.f;
        int   wasteStartLastTouch = -1;
        float wastePrevBallWallDist = 999999.f;

        // goal edge
        bool prevGoalScored = false;
        bool goalEdge = false;

        // no-bounce goal watch
        bool  nbWatch = false;
        bool  nbValid = false;
        float nbTimer = 0.f;
        float nbPrevBallWallDist = 999999.f;
    };

    static float Saturate01(float x) {
        if (x <= 0.f) return 0.f;
        if (x >= 1.f) return 1.f;
        return x;
    }

    static float Length2D(const Vec& v) {
        return std::sqrt(v.x*v.x + v.y*v.y);
    }

    float GetChainMult(int successCount) const {
        if (successCount <= 0) return mult1;
        if (successCount == 1) return mult2;
        if (successCount == 2) return mult3;
        if (successCount == 3) return mult4;
        return mult5Plus;
    }

    Vec GetOppGoalCenter(const Player& player) const {
        return (player.team == RocketSim::Team::BLUE)
            ? CommonValues::ORANGE_GOAL_CENTER
            : CommonValues::BLUE_GOAL_CENTER;
    }

    bool IsNearOppGoalXY(const Player& player, const GameState& state, float nearDist) const {
        const Vec oppGoal = GetOppGoalCenter(player);
        Vec d = state.ball.pos - oppGoal;
        d.z = 0.f;
        const float distXY = Length2D(d);
        return (distXY <= nearDist);
    }

    bool IsOppGoalScored(const Player& player, const GameState& state) const {
        const float oppGoalY = GetOppGoalCenter(player).y;
        return (state.ball.pos.y * oppGoalY) > 0.f;
    }

    bool IsInOwnGoalSideFracOfHalf(const Player& player, const GameState& state) const {
        const float Yb = CommonValues::BACK_WALL_Y;
        if (Yb <= 1e-3f) return false;

        // enemy side is + after sign flip
        const float signedY = (player.team == RocketSim::Team::BLUE) ? state.ball.pos.y : -state.ball.pos.y;

        // only own half
        if (signedY >= 0.f) return false;

        // frac=2/5 => region [-Yb, -(3/5)Yb]
        const float boundary = -(1.0f - ownGoalSideFracOfHalf) * Yb;
        return (signedY <= boundary);
    }

    float GetGoalAngleFactor(const Player& player, const GameState& state) const {
        const Vec oppGoal = GetOppGoalCenter(player);

        Vec goalDir = oppGoal - state.ball.pos;
        const float goalLen = goalDir.Length();
        if (goalLen <= 1e-3f) return 1.0f;
        goalDir /= goalLen;

        Vec vel = state.ball.vel;
        const float velLen = vel.Length();
        if (velLen <= 1e-3f) return 1.0f;
        vel /= velLen;

        float cosAng = std::clamp(vel.Dot(goalDir), -1.0f, 1.0f);
        const float angDeg = std::acos(cosAng) * (180.0f / 3.14159265f);

        if (angDeg <= angleStartDeg) return 1.0f;
        if (angDeg >= angleEndDeg) return angleMinFactor;

        float t = (angDeg - angleStartDeg) / (angleEndDeg - angleStartDeg);
        t = Saturate01(t);
        if (angleSmoothStep) t = t * t * (3.f - 2.f * t);

        const float factor = 1.0f + (angleMinFactor - 1.0f) * t;
        return std::clamp(factor, angleMinFactor, 1.0f);
    }

    float GetXySpeedFactor(const GameState& state) const {
        const float UU_PER_SEC_TO_KMH = 0.036f;

        const float vx = state.ball.vel.x;
        const float vy = state.ball.vel.y;

        const float vxy_uu  = std::sqrt(vx*vx + vy*vy);
        const float vxy_kmh = vxy_uu * UU_PER_SEC_TO_KMH;

        const float v = std::clamp(vxy_kmh, xyKmhMin, xyKmhMax);
        const float denom = std::max(1e-3f, (xyKmhMax - xyKmhMin));

        float t = (v - xyKmhMin) / denom;
        t = Saturate01(t);

        const float factor = xyMultMin + (xyMultMax - xyMultMin) * std::sqrt(t);
        return std::clamp(factor, std::min(xyMultMin, xyMultMax), std::max(xyMultMin, xyMultMax));
    }

    float GetHeightFactor(const Player& player, const GameState& state) const {
        if (IsNearOppGoalXY(player, state, disableHeightNearGoalXYDist)) {
            return 1.0f;
        }

        const float z = std::clamp(state.ball.pos.z, 0.f, CommonValues::CEILING_Z);

        const float zLow  = std::max(0.f, CommonValues::GOAL_HEIGHT * heightLowFracGoal);
        const float zHigh = std::max(zLow + 1.f, CommonValues::CEILING_Z * heightHighFracCeil);

        if (z <= zLow)  return heightMultAtLow;
        if (z >= zHigh) return heightMultAtHigh;

        const float t = (z - zLow) / (zHigh - zLow);
        const float mult = heightMultAtLow + (heightMultAtHigh - heightMultAtLow) * Saturate01(t);
        return std::max(0.f, mult);
    }

    float ComputeBallWallDist(const Vec& ballPos) const {
        float wallDistX = CommonValues::SIDE_WALL_X - std::abs(ballPos.x);
        float wallDistY = CommonValues::BACK_WALL_Y - std::abs(ballPos.y);
        float wallDist  = std::min(wallDistX, wallDistY);
        return std::max(0.f, wallDist);
    }

    float GetFieldPosMult(const Player& player, const GameState& state) const {
        const float ballDiam = 2.0f * CommonValues::BALL_RADIUS;
        const float nearOppGoalDist = enemyGoalCloseBallDiam * ballDiam;

        if (nearOppGoalDist > 0.f && IsNearOppGoalXY(player, state, nearOppGoalDist)) {
            return enemyGoalCloseMult;
        }

        const float Yb = CommonValues::BACK_WALL_Y;
        if (Yb <= 1e-3f) return 1.0f;

        const float signedY = (player.team == RocketSim::Team::BLUE) ? state.ball.pos.y : -state.ball.pos.y;

        // [-Yb, 0] : 0.3 -> 0.7
        if (signedY < 0.f) {
            float t = (signedY + Yb) / Yb;
            t = Saturate01(t);
            return ownBackboardMult + (midlineMult - ownBackboardMult) * t;
        }

        // [0, +Yb/2] : 0.7 -> 1.0
        const float yHalfMid = 0.5f * Yb;
        if (signedY <= yHalfMid && yHalfMid > 1e-3f) {
            float t = signedY / yHalfMid;
            t = Saturate01(t);
            return midlineMult + (enemyHalfMidlineMult - midlineMult) * t;
        }

        return enemyHalfMidlineMult;
    }

    // ===== No-bounce goal bonus =====
    void StartNoBounceGoalWatch(const GameState& state, PerCarState& s) const {
        s.nbWatch = true;
        s.nbValid = true;
        s.nbTimer = 0.f;
        s.nbPrevBallWallDist = ComputeBallWallDist(state.ball.pos);
    }

    float UpdateNoBounceGoalBonus(const Player& player, const GameState& state, float dt, PerCarState& s) const {
        if (!s.nbWatch) return 0.f;

        if (s.goalEdge) {
            float bonus = 0.f;
            if (s.nbValid && IsOppGoalScored(player, state)) {
                bonus = noBounceGoalBonus;
            }
            s.nbWatch = false;
            s.nbValid = false;
            s.nbTimer = 0.f;
            return bonus;
        }

        s.nbTimer += dt;
        if (s.nbTimer > noBounceMaxWatchSec) {
            s.nbWatch = false;
            s.nbValid = false;
            s.nbTimer = 0.f;
            return 0.f;
        }

        if (!s.nbValid) return 0.f;

        const float wallDistNow = ComputeBallWallDist(state.ball.pos);
        const bool wallTouchish =
            (wallDistNow <= noBounceWallTouchDist) &&
            (s.nbPrevBallWallDist > (noBounceWallTouchDist + noBounceWallTouchHyst));
        s.nbPrevBallWallDist = wallDistNow;

        if (wallTouchish) {
            s.nbValid = false;
            return 0.f;
        }

        const float ballDiam = 2.0f * CommonValues::BALL_RADIUS;
        const float zThr = noBounceGroundDiamMult * ballDiam;

        if (state.ball.pos.z <= zThr) {
            s.nbValid = false;
            return 0.f;
        }

        return 0.f;
    }

    // ===== wasted pop penalty =====
    void StartWastedPopWatch(const Player& player, const GameState& state, PerCarState& s) const {
        if (IsNearOppGoalXY(player, state, disableWastedPopNearGoalXYDist)) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return;
        }

        s.wasteWatch = true;
        s.wasteTimer = 0.f;

        s.wasteStartLastTouch = state.lastTouchCarID;
        s.wastePrevBallWallDist = ComputeBallWallDist(state.ball.pos);
    }

    float UpdateWastedPopPenalty(const Player& player, const GameState& state, float dt, PerCarState& s) const {
        if (!s.wasteWatch) return 0.f;

        if (IsNearOppGoalXY(player, state, disableWastedPopNearGoalXYDist) || state.goalScored) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return 0.f;
        }

        s.wasteTimer += dt;
        if (s.wasteTimer > wastedPopMaxWatchSec) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return 0.f;
        }

        if (state.lastTouchCarID != -1 && state.lastTouchCarID != s.wasteStartLastTouch) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return 0.f;
        }

        const float wallDistNow = ComputeBallWallDist(state.ball.pos);

        const bool wallTouchish =
            (wallDistNow <= ballWallTouchDist) &&
            (s.wastePrevBallWallDist > (ballWallTouchDist + ballWallTouchHyst));

        s.wastePrevBallWallDist = wallDistNow;

        if (wallTouchish) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return 0.f;
        }

        const float ballDiam = 2.0f * CommonValues::BALL_RADIUS;
        const float zThr = wastedPopGroundDiamMult * ballDiam;

        if (state.ball.pos.z <= zThr) {
            s.wasteWatch = false;
            s.wasteTimer = 0.f;
            return -wastedPopPenalty;
        }

        return 0.f;
    }

    bool IsRealGround(const Player& player, float dt, float& grace) const {
        float wallDistX = CommonValues::SIDE_WALL_X - std::abs(player.pos.x);
        float wallDistY = CommonValues::BACK_WALL_Y - std::abs(player.pos.y);
        float wallDist  = std::min(wallDistX, wallDistY);
        wallDist = std::max(0.f, wallDist);

        const bool vzOk = (std::abs(player.vel.z) <= maxAbsVzForGround);

        const bool lowZ     = (player.pos.z <= groundMaxCarZ);
        const bool nearWall = (wallDist <= wallNearDist) && (player.pos.z <= wallMaxCarZ);

        const bool groundNow = vzOk && (lowZ || nearWall);

        if (groundNow) grace = groundGraceSec;
        else grace = std::max(0.f, grace - dt);

        return (grace > 0.f);
    }

    // ===== params =====
    float scale;

    float flipAfterResetWindowSec;
    float maxArmedWaitSec;
    int   maxTouchAttempts;

    float minUpwardVelZGain;
    float minDeltaVNorm;

    float deltaVelScale;
    float zSqWeight;
    bool  onlyUpwardZ;
    float popSaturation;

    float mult1, mult2, mult3, mult4, mult5Plus;

    float groundMaxCarZ;
    float wallNearDist;
    float wallMaxCarZ;
    float maxAbsVzForGround;
    float groundGraceSec;

    float minAirZForReset;

    float angleStartDeg;
    float angleEndDeg;
    float angleMinFactor;
    bool  angleSmoothStep;

    float xyKmhMin;
    float xyKmhMax;
    float xyMultMin;
    float xyMultMax;

    float heightMultAtLow;
    float heightMultAtHigh;
    float heightLowFracGoal;
    float heightHighFracCeil;
    float disableHeightNearGoalXYDist;

    float wastedPopPenalty;
    float wastedPopMaxWatchSec;
    float wastedPopGroundDiamMult;
    float disableWastedPopNearGoalXYDist;
    float ballWallTouchDist;
    float ballWallTouchHyst;

    float ownBackboardMult;
    float midlineMult;
    float enemyHalfMidlineMult;
    float enemyGoalCloseMult;
    float enemyGoalCloseBallDiam;

    float noBounceGoalBonus;
    float noBounceMaxWatchSec;
    float noBounceGroundDiamMult;
    float noBounceWallTouchDist;
    float noBounceWallTouchHyst;
    int   noBounceMinPopCount;

    bool  penalizeSuccessInOwnGoalSideFrac;
    float ownGoalSideFracOfHalf;
    float ownGoalSideSuccessPenalty;

    std::unordered_map<uint32_t, PerCarState> states;
};


    


    class SimpleFlipResetLearnReward : public Reward {
public:
    // -------------------------
    // ????
    // -------------------------
    float scale = 1.0f;

    // -------------------------
    // ?????
    // -------------------------
    float minAirZ = 420.f;                 // ??????????????
    float maxBallDistForAcquire = 380.f;   // ?????????

    // ?/??/????+?????????????
    float ceilingMargin  = 300.f;
    float sideWallMargin = 280.f;
    float backWallMargin = 280.f;

    float cornerRadius = 1152.f; // ???????
    float cornerMargin = 280.f;

    float carInflate = 80.f;     // ???????????????????

    // -------------------------
    // ??
    // -------------------------
    float acquireBonus = 0.15f;
    float acquireCooldownSec = 0.20f;

    float shotScale = 2.3f;
    float minShotNorm = 0.10f;
    float zBoost = 1.6f;

    // -------------------------
    // 50/50???????????????
    // -------------------------
    bool  suppressShotAfterOppTouch = true;
    float suppressAfterOppTouchSec = 0.10f;

    // ???????????????????×4 ? 742uu?
    float oppNearDist = 2.f * 2.f * RLGC::CommonValues::BALL_RADIUS;

    // ??????????????????????????
    bool suppressAcquireWhenOppNear = true;

    // -------------------------
    // ????: ??????????????????
    // -------------------------
    float maxCarZ = 170.f;
    float maxAbsVzForFloor = 350.f;
    float floorGraceSec = 0.02f;

    // ? ????????????????????????
    SimpleFlipResetLearnReward(
        float scale = 1.0f,
        float minAirZ = 250.f,
        float maxBallDistForAcquire = 380.f,

        float ceilingMargin = 300.f,
        float sideWallMargin = 280.f,
        float backWallMargin = 280.f,
        float cornerRadius = 1152.f,
        float cornerMargin = 280.f,
        float carInflate = 80.f,

        float acquireBonus = 1.0f,
        float acquireCooldownSec = 0.20f,

        float shotScale = 2.0f,
        float minShotNorm = 0.10f,
        float zBoost = 1.6f,

        bool  suppressShotAfterOppTouch = true,
        float suppressAfterOppTouchSec = 0.10f,

        float oppNearDist = 4.f * 2.f * RLGC::CommonValues::BALL_RADIUS,
        bool  suppressAcquireWhenOppNear = true,

        float maxCarZ = 170.f,
        float maxAbsVzForFloor = 350.f,
        float floorGraceSec = 0.02f
    )
    : scale(scale),
      minAirZ(minAirZ),
      maxBallDistForAcquire(maxBallDistForAcquire),
      ceilingMargin(ceilingMargin),
      sideWallMargin(sideWallMargin),
      backWallMargin(backWallMargin),
      cornerRadius(cornerRadius),
      cornerMargin(cornerMargin),
      carInflate(carInflate),
      acquireBonus(acquireBonus),
      acquireCooldownSec(acquireCooldownSec),
      shotScale(shotScale),
      minShotNorm(minShotNorm),
      zBoost(zBoost),
      suppressShotAfterOppTouch(suppressShotAfterOppTouch),
      suppressAfterOppTouchSec(suppressAfterOppTouchSec),
      oppNearDist(oppNearDist),
      suppressAcquireWhenOppNear(suppressAcquireWhenOppNear),
      maxCarZ(maxCarZ),
      maxAbsVzForFloor(maxAbsVzForFloor),
      floorGraceSec(floorGraceSec)
    {}

private:
    struct Win {
        bool  active = false;   // ??????
        bool  used   = false;   // 2??????????????
        float cd     = 0.f;     // ????????
        float floorGrace = 0.f; // ??????????
    };

    std::unordered_map<uint32_t, Win> win;
    std::unordered_map<uint32_t, float> oppTouchSuppressTimer;

    inline const Player* GetPlayerByCarId(const GameState& s, int carId) const {
        for (const auto& p : s.players)
            if ((int)p.carId == carId) return &p;
        return nullptr;
    }

    inline bool TryGetTeamByCarId(const GameState& state, int carId, RocketSim::Team& outTeam) const {
        for (const auto& p : state.players) {
            if ((int)p.carId == carId) {
                outTeam = p.team;
                return true;
            }
        }
        return false;
    }

    inline bool IsOpponentTouchEventForPlayer(const GameState& state, const Player& player, int touchCarId) const {
        if (touchCarId < 0) return false;
        if (touchCarId == (int)player.carId) return false;

        RocketSim::Team t;
        if (!TryGetTeamByCarId(state, touchCarId, t)) return false;

        return t != player.team;
    }

    inline bool IsBlockedZone(const RocketSim::Vec& p) const {
        float ax = std::abs(p.x);
        float ay = std::abs(p.y);

        bool nearCeiling = (p.z > (RLGC::CommonValues::CEILING_Z - ceilingMargin - carInflate));
        bool nearSideWall = (ax > (RLGC::CommonValues::SIDE_WALL_X - sideWallMargin - carInflate));
        bool nearBackWall = (ay > (RLGC::CommonValues::BACK_WALL_Y - backWallMargin - carInflate));

        float cx = RLGC::CommonValues::SIDE_WALL_X - cornerRadius;
        float cy = RLGC::CommonValues::BACK_WALL_Y - cornerRadius;

        bool nearCornerArc = false;
        if (ax > cx && ay > cy) {
            float dx = ax - cx;
            float dy = ay - cy;
            float dist = std::sqrt(dx * dx + dy * dy);

            float thr = cornerRadius - cornerMargin - carInflate;
            nearCornerArc = (dist > thr);
        }

        return nearCeiling || nearSideWall || nearBackWall || nearCornerArc;
    }

    inline bool AnyOpponentNearBall(const GameState& s, const Player& me, float distThr) const {
        for (const auto& p : s.players) {
            if (p.team == me.team) continue;
            float d = (s.ball.pos - p.pos).Length();
            if (d <= distThr) return true;
        }
        return false;
    }

public:
    void Reset(const GameState&) override {
        win.clear();
        oppTouchSuppressTimer.clear();
    }

    float GetReward(const Player& player, const GameState& state, bool) override {
        if (!player.prev || !state.prev)
            return 0.f;

        float dt = state.deltaTime;
        Win& w = win[player.carId];

        // cooldown ??
        w.cd = std::max(0.f, w.cd - dt);

        // ?????????????
        float& oppTimer = oppTouchSuppressTimer[player.carId];
        oppTimer = std::max(0.f, oppTimer - dt);

        // --------------------------
        // ????isOnGround??????
        // --------------------------
        bool floorNow = (player.pos.z <= maxCarZ) && (std::abs(player.vel.z) <= maxAbsVzForFloor);
        if (floorNow) w.floorGrace = floorGraceSec;
        else          w.floorGrace = std::max(0.f, w.floorGrace - dt);

        bool onFloor = (w.floorGrace > 0.f);
        if (onFloor) {
            w.active = false;
            w.used   = false;
        }

        // --------------------------
        // ????????????????????
        // ? lastTouchCarID ???????????????
        // ? ????????????????????50/50???
        // --------------------------
        if (suppressShotAfterOppTouch && state.prev) {
            if (state.lastTouchCarID != state.prev->lastTouchCarID) {
                if (IsOpponentTouchEventForPlayer(state, player, state.lastTouchCarID)) {
                    const Player* opp = GetPlayerByCarId(state, state.lastTouchCarID);
                    if (opp) {
                        float dOppBall = (state.ball.pos - opp->pos).Length();
                        if (dOppBall <= oppNearDist) {
                            oppTimer = std::max(oppTimer, suppressAfterOppTouchSec);
                        }
                    }
                }
            }

            if (state.prev->prev && state.prev->lastTouchCarID != state.prev->prev->lastTouchCarID) {
                if (IsOpponentTouchEventForPlayer(*state.prev, player, state.prev->lastTouchCarID)) {
                    const Player* opp = GetPlayerByCarId(*state.prev, state.prev->lastTouchCarID);
                    if (opp) {
                        float dOppBall = (state.prev->ball.pos - opp->pos).Length();
                        if (dOppBall <= oppNearDist) {
                            oppTimer = std::max(oppTimer, suppressAfterOppTouchSec);
                        }
                    }
                }
            }
        }

        // --------------------------
        // ???????
        // --------------------------
        bool highNow  = (player.pos.z > minAirZ);
        bool highPrev = (player.prev->pos.z > minAirZ);

        bool blockedNow  = IsBlockedZone(player.pos);
        bool blockedPrev = IsBlockedZone(player.prev->pos);

        float ballDist = (state.ball.pos - player.pos).Length();
        bool nearBall = (ballDist <= maxBallDistForAcquire);

        // ??????????now/prev ????????????
        bool oppNearNow  = AnyOpponentNearBall(state, player, oppNearDist);
        bool oppNearPrev = AnyOpponentNearBall(*state.prev, player, oppNearDist);
        bool allowAcquire = !(suppressAcquireWhenOppNear && (oppNearNow || oppNearPrev));

        // 2??????????????=??????
        bool canNow  = player.HasFlipOrJump();
        bool canPrev = player.prev->HasFlipOrJump();

        float reward = 0.f;

        // --------------------------
        // 1) ???false->true??? + ?? + ????? + ?????????
        // --------------------------
        bool acquiredNow =
            (w.cd <= 0.f) &&
            allowAcquire &&
            highNow && highPrev &&
            nearBall &&
            (!blockedNow && !blockedPrev) &&
            (!canPrev && canNow);

        if (acquiredNow) {
            reward += acquireBonus;
            w.active = true;
            w.used   = false;
            w.cd     = acquireCooldownSec;
        }

        // --------------------------
        // 2) ??????? true->false
        // --------------------------
        if (w.active && !w.used) {
            bool usedNow = (canPrev && !canNow);
            if (usedNow) {
                w.used = true;
            }
        }

        // --------------------------
        // 3) ????????????? dv ?????1??
        //    ????????50/50???????
        // --------------------------
        if (w.used && player.ballTouchedStep) {
            bool suppressNow = suppressShotAfterOppTouch && (oppTimer > 0.f);

            if (!suppressNow) {
                RocketSim::Vec oppGoal = (player.team == RocketSim::Team::BLUE)
                    ? RLGC::CommonValues::ORANGE_GOAL_CENTER
                    : RLGC::CommonValues::BLUE_GOAL_CENTER;

                RocketSim::Vec toGoal = oppGoal - state.ball.pos;
                float toGoalLen = toGoal.Length();
                if (toGoalLen > 1e-3f) {
                    RocketSim::Vec dir = toGoal / toGoalLen;

                    RocketSim::Vec dv = state.ball.vel - state.prev->ball.vel;
                    dv.z *= zBoost;

                    float vToward = dv.Dot(dir);
                    float norm = std::clamp(vToward / RLGC::CommonValues::BALL_MAX_SPEED, 0.f, 1.f);

                    if (norm >= minShotNorm) {
                        reward += shotScale * norm;
                    }
                }
            }

            // ??????????????
            w.active = false;
            w.used   = false;
        }

        return reward * scale;
    }
};



	class SimpleResetTeachReward : public Reward {
	public:
		float proximityScale;   // Reward for being close to the ball
		float maxProximityDist; // UU, beyond this proximity reward is 0
		float alignmentScale;   // Reward for wheels facing the ball
		float resetBonus;       // One-time reward when flip reset is gained

		SimpleResetTeachReward(
			float proximityScale   = 0.3f,
			float maxProximityDist = 600.f,
			float alignmentScale   = 0.5f,
			float resetBonus       = 5.f
		) : proximityScale(proximityScale),
		    maxProximityDist(maxProximityDist),
		    alignmentScale(alignmentScale),
		    resetBonus(resetBonus) {
		}

		virtual void Reset(const GameState& state) override {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0.f;

			// Only reward while in the air
			if (player.isOnGround) return 0.f;

			float reward = 0.f;

			Vec   dirToBall = state.ball.pos - player.pos;
			float dist      = dirToBall.Length();

			if (dist > 1.f) {
				dirToBall = dirToBall / dist;

				// 1. Proximity: 1 when touching, 0 at maxProximityDist
				float proximity = 1.f - RS_CLAMP(dist / maxProximityDist, 0.f, 1.f);
				reward += proximityScale * proximity;

				// 2. Wheel alignment: -rotMat.up is the underside/wheels direction
				Vec   underside  = player.rotMat.up * -1.f;
				float alignment  = underside.Dot(dirToBall); // -1..1

				if (alignment > 0.f)
					reward += alignmentScale * alignment * proximity; // gated by proximity
			}

			// 3. Reset bonus: fires the moment the flip reset is gained FROM THE BALL.
			// Gated on ballTouchedStep so wall contact doesn't trigger this.
			bool gainedReset = !player.prev->HasFlipReset() && player.HasFlipReset() && player.ballTouchedStep;
			if (gainedReset) {
				std::cout << "SimpleResetTeachReward: reset gained! +" << resetBonus << std::endl;
				reward += resetBonus;
			}

			return reward;
		}
	};


class MaktufResetReward : public Reward {
	public:
		// How rolled the car must be for the flip to count as a maktuf flip.
		// rotMat.up.z ranges -1 (upside down) to 1 (right side up).
		// At 0 the car is perfectly on its side. 0.5 gives ~60 degrees of roll,
		// which is a generous but fair threshold.
		float maxUpZ;

		// After the sideways flip is done, how many steps to wait for the
		// cancel + reset to happen. At 8-tick skip, 30 steps ~= 2 seconds.
		// FLIP_TORQUE_TIME is 0.65s, so we need at least that plus some grace.
		int maxStepsAfterFlip;

		MaktufResetReward(
			float maxUpZ           = 0.5f,
			int   maxStepsAfterFlip = 30
		) : maxUpZ(maxUpZ),
		    maxStepsAfterFlip(maxStepsAfterFlip) {
		}

		virtual void Reset(const GameState& state) override {
			stepsSinceFlip.clear();
			chainCount.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0.f;

			auto id = player.carId;

			// Landing breaks the chain and kills any active window
			if (player.isOnGround) {
				if (chainCount[id] > 0)
					std::cout << "MaktufResetReward: chain of " << chainCount[id] << " broken by landing" << std::endl;
				stepsSinceFlip[id] = -1;
				chainCount[id]     = 0;
				return 0.f;
			}

			bool prevHadFlipReset = player.prev->HasFlipReset();
			bool nowHasFlipReset  = player.HasFlipReset();

			// ── Stage 1+2: Flip initiated while sideways ──────────────────────
			// hasFlipped just became true (flip was started this step)
			bool flipJustStarted = player.hasFlipped && !player.prev->hasFlipped;

			if (flipJustStarted) {
				// Check sideways orientation at the moment of the flip
				float upZ = player.rotMat.up.z;
				if (std::abs(upZ) < maxUpZ) {
					// Valid maktuf flip - open the window
					stepsSinceFlip[id] = 0;
				} else {
					// Flip happened but car wasn't sideways - kill any existing window
					stepsSinceFlip[id] = -1;
				}
			}

			// Tick the window if active
			int& steps = stepsSinceFlip[id];
			if (steps >= 0)
				steps++;

			// Window expired - they took too long to align with the ball
			if (steps > maxStepsAfterFlip) {
				if (chainCount[id] > 0)
					std::cout << "MaktufResetReward: chain of " << chainCount[id] << " broken by timeout" << std::endl;
				steps          = -1;
				chainCount[id] = 0;
				return 0.f;
			}

			// ── Stage 3+4: Flip cancelled and reset gained ────────────────────
			// HasFlipReset() just became true: ball contact gave back the flip
			bool gainedReset = !prevHadFlipReset && nowHasFlipReset;

			if (steps >= 0 && gainedReset) {
				// Confirm that a flip was actually performed (not just a double jump)
				// player.hasFlipped being false here is expected - the ball reset it.
				// We know a flip was done because the window was opened in Stage 1+2.
				steps = -1; // Close the window

				int& chain = chainCount[id];
				chain++;

				float reward = 1.f;
				for (int i = 1; i < chain; i++)
					reward *= 3.f;

				std::cout << "MaktufResetReward: chain x" << chain
				          << " -> " << reward
				          << " (upZ at flip=" << player.rotMat.up.z << ")" << std::endl;
				return reward;
			}

			return 0.f;
		}

	private:
		// -1 = no active window, >= 0 = steps since the sideways flip was initiated
		std::map<uint32_t, int> stepsSinceFlip;

		// How many consecutive maktuf resets in this aerial
		std::map<uint32_t, int> chainCount;
	};


class MaktufResetRewardContinuous : public Reward {
	public:
		// ── Stage 0 ──────────────────────────────────────────────────────────
		// Scale for the sideways-orientation reward (given every step in the air)
		float sidewaysScale;

		// ── Stage 1 ──────────────────────────────────────────────────────────
		// How rolled the car must be for the flip to open Stage 1.
		// abs(rotMat.up.z) < maxUpZ to qualify. 0.5 = ~60 degrees of roll.
		float maxUpZ;

		// Steps after the sideways flip to keep looking for the reset.
		// At 8-tick skip, 30 steps ~= 2 seconds.
		int maxStepsAfterFlip;

		// Scale for the underside-toward-ball alignment reward
		float alignmentScale;

		// Scale for the proximity reward (only given when alignment > 0)
		// Proximity is normalised: 1 - clamp(dist / maxProximityDist, 0, 1)
		float proximityScale;
		float maxProximityDist; // UU, beyond this proximity reward is 0

		// ── Stage 2 ──────────────────────────────────────────────────────────
		// One-time bonus when the reset is actually gained
		float resetBonus;

		MaktufResetRewardContinuous(
			float sidewaysScale    = 0.1f,
			float maxUpZ           = 0.5f,
			int   maxStepsAfterFlip = 30,
			float alignmentScale   = 0.5f,
			float proximityScale   = 0.3f,
			float maxProximityDist = 500.f,
			float resetBonus       = 3.f
		) : sidewaysScale(sidewaysScale),
		    maxUpZ(maxUpZ),
		    maxStepsAfterFlip(maxStepsAfterFlip),
		    alignmentScale(alignmentScale),
		    proximityScale(proximityScale),
		    maxProximityDist(maxProximityDist),
		    resetBonus(resetBonus) {
		}

		virtual void Reset(const GameState& state) override {
			stepsSinceFlip.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0.f;

			auto id = player.carId;

			if (player.isOnGround) {
				stepsSinceFlip[id] = -1;
				return 0.f;
			}

			float reward = 0.f;

			bool prevHadFlipReset = player.prev->HasFlipReset();
			bool nowHasFlipReset  = player.HasFlipReset();
			bool gainedReset      = !prevHadFlipReset && nowHasFlipReset;

			// ── Stage 0: Sideways reward ──────────────────────────────────────
			// Always given while in the air to encourage rolling onto the side.
			// 1 - abs(up.z): 0 when upright, 1 when perfectly sideways.
			float sidewaysness = 1.f - std::abs(player.rotMat.up.z);
			reward += sidewaysScale * sidewaysness;

			// ── Open Stage 1 window on a sideways flip ────────────────────────
			bool flipJustStarted = player.hasFlipped && !player.prev->hasFlipped;
			if (flipJustStarted) {
				if (std::abs(player.rotMat.up.z) < maxUpZ)
					stepsSinceFlip[id] = 0;  // Valid maktuf flip - open window
				else
					stepsSinceFlip[id] = -1; // Flip wasn't sideways - close window
			}

			// Tick window
			int& steps = stepsSinceFlip[id];
			if (steps >= 0)
				steps++;

			// Expire window
			if (steps > maxStepsAfterFlip)
				steps = -1;

			// ── Stage 1: Alignment + proximity rewards ────────────────────────
			// Only given while the Stage 1 window is open (after a sideways flip)
			if (steps >= 0) {
				Vec  dirToBall  = (state.ball.pos - player.pos);
				float dist      = dirToBall.Length();

				if (dist > 1.f) { // Avoid divide-by-zero
					dirToBall = dirToBall / dist;

					// -rotMat.up is the underside of the car (belly/wheels direction)
					Vec  underside  = player.rotMat.up * -1.f;
					float alignment = underside.Dot(dirToBall); // -1..1, 1 = perfect

					if (alignment > 0.f) {
						// Alignment reward - scales with how well belly faces ball
						reward += alignmentScale * alignment;

						// Proximity reward - only meaningful when aligned
						float proxNorm = 1.f - RS_CLAMP(dist / maxProximityDist, 0.f, 1.f);
						reward += proximityScale * proxNorm * alignment;
					}
				}

				// ── Stage 2: Reset bonus ──────────────────────────────────────
				if (gainedReset) {
					steps = -1;
					std::cout << "MaktufResetRewardContinuous: reset gained! bonus=" << resetBonus << std::endl;
					reward += resetBonus;
				}
			}

			return reward;
		}

	private:
		// -1 = no active window, >= 0 = steps since sideways flip was initiated
		std::map<uint32_t, int> stepsSinceFlip;
	};

	
	
class RipplesFlipResetFreestyleChain : public Reward {
public:
	// Dense approach reward
	float approach_weight = 1.0f;
	float min_height = 300.0f;
	float max_approach_dist = 500.0f;
	float min_roof_up = 0.6f;
	float min_approach_speed = 150.0f;

	// Reset rewards (escalating)
	float base_reset_reward = 10.0f;
	float chain_multiplier = 1.5f;
	float max_chain_multiplier = 5.0f;
	int   max_chain_count = 4;              // hard cap on resets per aerial
	uint64_t min_ticks_between_resets = 90;  // cooldown — ignore resets too close together

	// Air dribble / sustained control between resets
	float air_control_reward = 0.3f;
	float max_air_control_dist = 250.0f;

	// Flip usage reward
	float flip_shot_reward = 20.0f;
	float chain_flip_bonus = 10.0f;
	float goal_align_bonus = 15.0f;
	float max_ticks_to_use = 300.f;

	// Direction scaling
	bool use_direction_scaling = true;

	// State tracking
	struct PlayerState {
		int   chain_count = 0;
		bool  has_flip_available = false;
		uint64_t last_reset_tick = 0;
		float last_dist_to_ball = 99999.f;
		bool  was_near_ball = false;
	};
	std::unordered_map<uint32_t, PlayerState> player_states;

	std::string GetName() override { return "RipplesFlipResetFreestyleChain"; }

	void Reset(const GameState& initialState) override {
		player_states.clear();
	}

	bool HasFlip(const Player& p) {
		return !p.hasDoubleJumped && !p.hasFlipped;
	}

	bool HasJump(const Player& p) {
		return !p.hasJumped;
	}

	bool HasFlipOrJump(const Player& p) {
		return HasFlip(p) || HasJump(p);
	}

	float ChainScale(int chain_count) {
		float scale = std::pow(chain_multiplier, static_cast<float>(chain_count));
		return std::min(scale, max_chain_multiplier);
	}

	float GetReward(const Player& player, const GameState& state, bool isFinal) override {
		if (!state.lastArena) return 0.f;

		uint64_t currentTick = state.lastArena->tickCount;
		auto& ps = player_states[player.carId];
		float reward = 0.0f;

		float car_height = player.pos.z;
		Vec down = -player.rotMat.up;
		Vec ball_pos = state.ball.pos;
		Vec player_pos = player.pos;

		bool isBlue = (player.team == Team::BLUE);
		float targetGoalY = isBlue ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;

		float dist = (player_pos - ball_pos).Length();
		dist = std::max(0.0f, dist - CommonValues::BALL_RADIUS);
		float dt = state.deltaTime > 1e-6f ? state.deltaTime : (1.0f / 120.0f);
		float approach_speed = (ps.last_dist_to_ball - dist) / dt;

		// =================================================================
		// LANDED = FULL CHAIN RESET
		// =================================================================
		if (player.isOnGround && car_height < min_height) {
			ps.chain_count = 0;
			ps.has_flip_available = false;
			ps.was_near_ball = false;
			ps.last_dist_to_ball = dist;
			return 0.0f;
		}

		if (car_height < min_height) {
			ps.last_dist_to_ball = dist;
			return 0.0f;
		}

		bool below_ball = player_pos.z < ball_pos.z;

		// Direction scaling
		float direction_scale = 1.0f;
		if (use_direction_scaling) {
			Vec opp_goal = (player.team == Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;
			Vec own_goal = (player.team == Team::BLUE)
				? CommonValues::BLUE_GOAL_CENTER
				: CommonValues::ORANGE_GOAL_CENTER;

			Vec ball_to_opp = (opp_goal - ball_pos).Normalized();
			Vec ball_to_own = (own_goal - ball_pos).Normalized();
			Vec player_vel_norm = player.vel.Normalized();

			float opp_dot = ball_to_opp.Dot(player_vel_norm);
			float own_dot = ball_to_own.Dot(player_vel_norm);
			direction_scale = 0.5f + 0.25f * (opp_dot - own_dot);
			direction_scale = std::max(0.2f, direction_scale);
		}

		// =================================================================
		// AIR CONTROL REWARD (between resets)
		// =================================================================
		if (car_height > min_height && ball_pos.z > min_height && dist < max_air_control_dist) {
			float proximity = 1.0f - (dist / max_air_control_dist);
			float height_bonus = std::min((car_height - min_height) / 500.0f, 1.0f);
			float chain_bonus = 1.0f + 0.3f * ps.chain_count;
			reward += air_control_reward * proximity * height_bonus * chain_bonus * direction_scale;
			ps.was_near_ball = true;
		} else {
			ps.was_near_ball = false;
		}

		// =================================================================
		// CONTINUOUS APPROACH
		// =================================================================
		if (!ps.has_flip_available &&
			below_ball &&
			ball_pos.z > min_height &&
			dist < max_approach_dist &&
			down.z > min_roof_up &&
			approach_speed >= min_approach_speed) {

			Vec player_to_ball = (ball_pos - player_pos).Normalized();
			float cossim = player_to_ball.Dot(down);
			float dist_factor = 1.0f - (dist / max_approach_dist);
			float speed_factor = std::min(approach_speed / (min_approach_speed * 3.0f), 1.5f);
			float chain_approach = 1.0f + 0.2f * ps.chain_count;

			float approach_reward = approach_weight * cossim * dist_factor * direction_scale * speed_factor * chain_approach;
			reward += std::max(0.0f, approach_reward);
		}

		// =================================================================
		// RESET DETECTION (capped at 4, 90-tick cooldown)
		// =================================================================
		if (state.prev && !ps.has_flip_available && ps.chain_count < max_chain_count) {
			const Player* prev = nullptr;
			for (const auto& p : state.prev->players) {
				if (p.carId == player.carId) {
					prev = &p;
					break;
				}
			}

			if (prev) {
				bool was_in_air = !prev->isOnGround;
				bool still_in_air = !player.isOnGround;
				bool got_flip_back = (HasFlip(player) && !HasFlip(*prev)) ||
				                     (HasJump(player) && !HasJump(*prev));

				// Cooldown check (skip on first reset where last_reset_tick == 0)
				bool cooldown_ok = (ps.last_reset_tick == 0) ||
				                   (currentTick - ps.last_reset_tick >= min_ticks_between_resets);

				if (was_in_air && still_in_air && got_flip_back && cooldown_ok) {
					if (ball_pos.z > min_height &&
						car_height > min_height &&
						dist < 200.0f &&
						down.z > 0.5f) {

						float scale = ChainScale(ps.chain_count);
						reward += base_reset_reward * scale * direction_scale;

						ps.chain_count++;
						ps.has_flip_available = true;
						ps.last_reset_tick = currentTick;
					}
				}
			}
		}

		// =================================================================
		// FLIP SHOT / FLIP USAGE
		// =================================================================
		if (ps.has_flip_available && state.prev) {
			const Player* prev = nullptr;
			for (const auto& p : state.prev->players) {
				if (p.carId == player.carId) {
					prev = &p;
					break;
				}
			}

			if (prev) {
				bool used_flip = HasFlipOrJump(*prev) && !HasFlipOrJump(player);
				bool touched_ball = player.ballTouchedStep;
				bool in_air = !player.isOnGround;
				uint64_t ticks_since_reset = currentTick - ps.last_reset_tick;

				if (used_flip && in_air && ticks_since_reset <= max_ticks_to_use) {
					if (touched_ball) {
						Vec goalCenter(0.f, targetGoalY, CommonValues::GOAL_HEIGHT / 2.f);
						Vec ballToGoal = (goalCenter - ball_pos).Normalized();
						float goalAlign = std::max(0.f, state.ball.vel.Normalized().Dot(ballToGoal));

						float chain_bonus_val = chain_flip_bonus * static_cast<float>(ps.chain_count);
						reward += flip_shot_reward + chain_bonus_val + goal_align_bonus * goalAlign;
					}
					ps.has_flip_available = false;
				}
				else if (ticks_since_reset > max_ticks_to_use) {
					ps.has_flip_available = false;
				}
			}
		}

		ps.last_dist_to_ball = dist;
		return reward;
	}
};
	
	
	
	
	
	
	
	// ============================================================================
// VELOCITY BALL TO GOAL MOUTH REWARD - Targets goal center, not corners
// Same weighting as VelocityBallToGoalReward
// ============================================================================
class VelocityBallToGoalMouthReward : public Reward {
public:
    bool ownGoal = false;
    
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
        bool targetOrangeGoal = player.team == Team::BLUE;
        if (ownGoal) targetOrangeGoal = !targetOrangeGoal;
        
        // Target center of goal mouth instead of back corner
        Vec targetPos = targetOrangeGoal
            ? Vec(0.f, CommonValues::BACK_WALL_Y, CommonValues::GOAL_HEIGHT / 2.f)
            : Vec(0.f, -CommonValues::BACK_WALL_Y, CommonValues::GOAL_HEIGHT / 2.f);
        
        Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
        return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
    }
};




	class DribbleAirdribbleBumpDemoRewardv2 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			float reward = 0.f;
			if (state.ball.pos.y > 4600 && std::abs(state.ball.pos.x) < GOAL_WIDTH_FROM_CENTER && state.ball.vel.y > 0) { // if the ball is in front of the opponents net and going towards it
				if (player.pos.y > state.ball.pos.y && std::abs(player.pos.x) < GOAL_WIDTH_FROM_CENTER) { // if the player is in front of the ball and in front of the net
					reward += 0.001f; // With the reward being 2000 atm, this should reward it 2 per step, or 60 per second. (this might be to high)
					if (player.eventState.bump) { // if the player bumped the opponent
						reward += 0.5f * (state.ball.vel.y / 1000); // 6000 is max ball speed but will rarely be that fast. scaling by 1000 should be good because player speed averages around 1400
					}
					if (player.eventState.demo) { // if the player demoed the opponent
						reward += 1.0f * (state.ball.vel.y / 1000); // this scales it by the speed of the ball towards the net / opponents side
					}
				}
			}
			return reward;
		}
	};

	class OffensivedemoRewardV1 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			float reward = 0.f;

			const float minDemoSpeed = 1600.f;    // must be supersonicish to demo
			const float minBumpSpeed = 1200.f;    // minimum speed to attempt bump
			const float ballBehindTolerance = 200.f;

			bool ballInFrontOfEnemyNet =
				state.ball.pos.y > 4600 &&
				std::abs(state.ball.pos.x) < GOAL_WIDTH_FROM_CENTER &&
				state.ball.vel.y > 0;

			bool playerInFrontOfBall =
				player.pos.y > state.ball.pos.y + ballBehindTolerance &&
				std::abs(player.pos.x) < GOAL_WIDTH_FROM_CENTER;

			bool ballBehindPlayer =
				state.ball.pos.y < player.pos.y - ballBehindTolerance;

			float playerSpeed = player.vel.Length();

			// -----------------------------------------------------------
			// 🔥 1. Strong reward for correct dribble-bump setup:
			// -----------------------------------------------------------
			if (ballInFrontOfEnemyNet && playerInFrontOfBall)
			{
				// BIGGER reward for positioning in front of ball
				reward += 0.01f;

				// Must drop ball behind the car to set up the bump
				if (ballBehindPlayer)
				{
					reward += 0.05f;   // encourages setup

					// Must go FAST to actually bump or demo
					if (playerSpeed > minBumpSpeed)
					{
						reward += 0.02f;

						// If bump happened
						if (player.eventState.bump)
						{
							reward += 1.0f * (state.ball.vel.y / 1000.f);
						}

						// If demo happened
						if (player.eventState.demo)
						{
							reward += 2.0f * (state.ball.vel.y / 800.f);
						}
					}
				}
			}

			// -----------------------------------------------------------
			// 🔥 2. Offensive demos even when NOT dribble bumping
			// -----------------------------------------------------------
			// Player is on offense if ball is in opponent half
			bool onOffense = state.ball.pos.y > 0;

			if (onOffense)
			{
				// If player is supersonic → reward demo chasing
				if (playerSpeed > minDemoSpeed)
				{
					reward += 0.005f;
				}

				// Demo reward
				if (player.eventState.demo)
				{
					reward += 1.5f;
				}

				// Missed demo attempt → penalize
				if (playerSpeed > minDemoSpeed && !player.eventState.demo)
				{
					reward -= 0.003f;
				}
			}

			return reward;
		}
	};


	class DribbleAirdribbleBumpDemoRewardv3 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			float reward = 0.f;

			// --- CONDITIONS FOR BEING IN A SCORING POSITION ---
			bool ballInFrontOfGoal =
				state.ball.pos.y > 4600 &&
				std::abs(state.ball.pos.x) < GOAL_WIDTH_FROM_CENTER &&
				state.ball.vel.y > 300;                    // Require meaningful speed toward net

			bool playerInFrontOfBall =
				player.pos.y > state.ball.pos.y &&
				std::abs(player.pos.x) < GOAL_WIDTH_FROM_CENTER;

			if (ballInFrontOfGoal && playerInFrontOfBall)
			{
				// --- REWARD POSITIONING STRONGLY ---
				reward += 0.05f; // Much higher reward for being properly positioned

				// --- SPEED REQUIREMENT FOR BUMPS / DEMOS ---
				float playerSpeed = player.vel.Length();     // You must go fast for demo chances
				bool fastEnoughForDemo = playerSpeed > 1300; // Near supersonic threshold

				// --- BUMP REWARD ---
				if (player.eventState.bump && fastEnoughForDemo)
				{
					// Scaled by how FAST the ball is going toward the net (more valuable bump)
					float ballSpeedTowardNet = state.ball.vel.y / 2000.f;
					reward += 1.0f * ballSpeedTowardNet;
				}

				// --- DEMO REWARD ---
				if (player.eventState.demo && fastEnoughForDemo)
				{
					// Higher scaling for demos; speed gives more reward
					float ballSpeedTowardNet = state.ball.vel.y / 2000.f;
					reward += 3.0f * ballSpeedTowardNet; // Much bigger reward for demos
				}
			}

			return reward;
		}
	};
	
	
	
	
	
	
class RipplesFlipResetRewardSpeedV2 : public Reward {
	public:
		// Dense approach reward (only fires when NO reset yet)
		float approach_weight = 1.0f;
		float min_height = 300.0f;
		float max_approach_dist = 500.0f;
		float min_roof_up = 0.7f;
		float min_approach_speed = 200.0f;
		
		// Event reward
		float reset_reward = 10.0f;
		bool  use_direction_scaling = true;
		
		// Flip shot reward
		float flip_shot_reward = 30.0f;
		float goal_align_bonus = 15.0f;
		float max_ticks_to_use = 300.f;
		
		// State tracking
		struct PlayerState {
			bool has_flipreset = false;
			uint64_t reset_tick = 0;
			float last_dist_to_ball = 99999.f;
		};
		std::unordered_map<uint32_t, PlayerState> player_states;

		std::string GetName() override { return "RipplesFlipResetRewardSpeedV2"; }

		void Reset(const GameState& initialState) override {
			player_states.clear();
		}

		bool HasFlip(const Player& p) {
			return !p.hasDoubleJumped && !p.hasFlipped;
		}

		bool HasJump(const Player& p) {
			return !p.hasJumped;
		}

		bool HasFlipOrJump(const Player& p) {
			return HasFlip(p) || HasJump(p);
		}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.lastArena) return 0.f;
			
			uint64_t currentTick = state.lastArena->tickCount;
			auto& ps = player_states[player.carId];
			float reward = 0.0f;

			float car_height = player.pos.z;
			Vec down = -player.rotMat.up;
			Vec ball_pos = state.ball.pos;
			Vec player_pos = player.pos;
			
			bool isBlue = (player.team == Team::BLUE);
			float targetGoalY = isBlue ? CommonValues::BACK_WALL_Y : -CommonValues::BACK_WALL_Y;

			float dist = (player_pos - ball_pos).Length();
			dist = std::max(0.0f, dist - CommonValues::BALL_RADIUS);
			float dt = state.deltaTime > 1e-6f ? state.deltaTime : (1.0f / 120.0f);
			float approach_speed = (ps.last_dist_to_ball - dist) / dt;

			// =================================================================
			// LANDED = RESET CYCLE
			// =================================================================
			if (player.isOnGround && car_height < min_height) {
				ps.has_flipreset = false;
				ps.last_dist_to_ball = dist;
				return 0.0f;
			}

			// Must be in air above min height
			if (car_height < min_height) {
				ps.last_dist_to_ball = dist;
				return 0.0f;
			}

			// Must be below ball for approach/reset
			bool below_ball = player_pos.z < ball_pos.z;

			// Direction scaling
			float direction_scale = 1.0f;
			if (use_direction_scaling) {
				Vec opp_goal = (player.team == Team::BLUE) 
					? CommonValues::ORANGE_GOAL_CENTER 
					: CommonValues::BLUE_GOAL_CENTER;
				Vec own_goal = (player.team == Team::BLUE) 
					? CommonValues::BLUE_GOAL_CENTER 
					: CommonValues::ORANGE_GOAL_CENTER;
				
				Vec ball_to_opp = (opp_goal - ball_pos).Normalized();
				Vec ball_to_own = (own_goal - ball_pos).Normalized();
				Vec player_vel_norm = player.vel.Normalized();
				
				float opp_dot = ball_to_opp.Dot(player_vel_norm);
				float own_dot = ball_to_own.Dot(player_vel_norm);
				direction_scale = 0.5f + 0.25f * (opp_dot - own_dot);
				direction_scale = std::max(0.2f, direction_scale);
			}

			// =================================================================
			// CONTINUOUS APPROACH (ONLY IF NO RESET YET)
			// =================================================================
			if (!ps.has_flipreset && 
				below_ball && 
				ball_pos.z > min_height &&
				dist < max_approach_dist && 
				down.z > min_roof_up && 
				approach_speed >= min_approach_speed) {
				
				Vec player_to_ball = (ball_pos - player_pos).Normalized();
				float cossim = player_to_ball.Dot(down);
				float dist_factor = 1.0f - (dist / max_approach_dist);
				float speed_factor = std::min(approach_speed / (min_approach_speed * 3.0f), 1.5f);
				
				float approach_reward = approach_weight * cossim * dist_factor * direction_scale * speed_factor;
				reward += std::max(0.0f, approach_reward);
			}

			// =================================================================
			// RESET DETECTION (ONLY IF NO RESET YET)
			// =================================================================
			if (state.prev && !ps.has_flipreset) {
				const Player* prev = nullptr;
				for (const auto& p : state.prev->players) {
					if (p.carId == player.carId) {
						prev = &p;
						break;
					}
				}
				
				if (prev) {
					bool was_in_air = !prev->isOnGround;
					bool still_in_air = !player.isOnGround;
					bool got_flip_back = (HasFlip(player) && !HasFlip(*prev)) ||
					                     (HasJump(player) && !HasJump(*prev));
					
					if (was_in_air && still_in_air && got_flip_back) {
						if (ball_pos.z > min_height && 
							car_height > min_height &&
							dist < 200.0f && 
							down.z > 0.5f) {
							
							reward += reset_reward * direction_scale;
							ps.has_flipreset = true;
							ps.reset_tick = currentTick;
						}
					}
				}
			}

			// =================================================================
			// FLIP SHOT (ONLY IF HAS RESET)
			// =================================================================
			if (ps.has_flipreset && state.prev) {
				const Player* prev = nullptr;
				for (const auto& p : state.prev->players) {
					if (p.carId == player.carId) {
						prev = &p;
						break;
					}
				}
				
				if (prev) {
					bool used_flip = HasFlipOrJump(*prev) && !HasFlipOrJump(player);
					bool touched_ball = player.ballTouchedStep;
					bool in_air = !player.isOnGround;
					uint64_t ticks_since_reset = currentTick - ps.reset_tick;
					
					if (used_flip && touched_ball && in_air && ticks_since_reset <= max_ticks_to_use) {
						Vec goalCenter(0.f, targetGoalY, CommonValues::GOAL_HEIGHT / 2.f);
						Vec ballToGoal = (goalCenter - ball_pos).Normalized();
						float goalAlign = std::max(0.f, state.ball.vel.Normalized().Dot(ballToGoal));
						
						reward += flip_shot_reward + goal_align_bonus * goalAlign;
						ps.has_flipreset = false;
					}
					// Timeout
					else if (ticks_since_reset > max_ticks_to_use) {
						ps.has_flipreset = false;
					}
				}
			}

			ps.last_dist_to_ball = dist;
			return reward;
		}
	};
	
	
	
	
// ============================================================================
// ConsolidatedDemoReward
// ============================================================================
// REPLACES ALL OF THESE:
//   - DemoReward (event-only, no shaping)
//   - BumpReward (event-only, no shaping)
//   - ShieldzDBRew (event-only, rewardScale=20000 nuke)
//   - StrategicDemoReward (event-only, capped at 1.5)
//   - DribbleAirdribbleBumpDemoRewardv2 (Blue-only bug, demo mult = 0.0)
//
// WHY THE OLD SETUP DIDN'T WORK:
//   The bot never demos anyone accidentally, so it never sees the 1300 reward.
//   0 * 1300 = 0.  Meanwhile VelPlayerToBall prints ~70/sec for chasing ball.
//   The bot has no incentive to STOP chasing ball to START chasing a player.
//
// HOW THIS FIXES IT:
//   Dense continuous shaping (bread-crumb trail) that rewards the bot for
//   moving toward opponents at speed. These small continuous rewards compete
//   with ball-chasing rewards, so the bot learns "sometimes leaving the ball
//   is profitable." Once it starts bumping/demoing by following the trail,
//   the moderate event rewards reinforce the behavior.
//
// RECOMMENDED USAGE:
//   { new ZeroSumReward(new ConsolidatedDemoReward(), 0.5f, 1.0f), 100.f },
//
//   That's it. One line. Delete the other 5 demo rewards.
//   Adjust weight 80-150 depending on how demo-heavy you want the bot.
//
// REWARD BUDGET AT WEIGHT 100, ZS(0.5, 1.0):
//   Approach shaping:     ~3/step  = ~60/sec  (competitive with ball-chase)
//   Bump event:           ~150     (moderate, ~12% of a goal)
//   Demo event:           ~300     (meaningful, ~23% of a goal)
//   Demo near goal+ball:  ~500     (juicy, ~38% of a goal)
//   Dribble-bump combo:   ~450     (sick play, ~35% of a goal)
//   Dribble-demo combo:   ~700     (clip-worthy, ~54% of a goal)
//
// ============================================================================

class ConsolidatedDemoReward : public Reward {
public:

    // ====================================================================
    // CONTINUOUS SHAPING — the bread-crumb trail that makes this learnable
    // ====================================================================

    // Approach: reward for velocity aimed at nearest opponent while fast
    float approachWeight     = 0.00f;
    float approachSpeedMin   = 1400.f;  // Start rewarding below supersonic to encourage building speed

    // Targeting: reward for facing an opponent while supersonic
    float targetWeight       = 0.005f;

    // Closing: reward for shrinking the gap quickly
    float closingWeight      = 0.000f;

    // Max distance to consider an opponent as a target
    float maxTargetDist      = 3500.f;

    // ====================================================================
    // EVENT REWARDS — moderate spikes, not galaxy-brained
    // ====================================================================
    float bumpBase           = 0.50f;   // Base bump reward
    float demoBase           = 1.00f;   // Base demo reward

    // ====================================================================
    // CONTEXT MULTIPLIERS — what makes a demo smart vs. pointless
    // ====================================================================

    // Ball proximity: demoing someone near the ball is more valuable
    float ballProxRadius     = 2500.f;  // Opponent within this distance of ball gets bonus
    float ballProxMaxBonus   = 0.60f;   // Max extra multiplier from ball proximity

    // Goal front: demoing a defender in front of their goal is huge
    float goalFrontRadius    = 3000.f;  // Distance from opponent's goal center
    float goalFrontMaxBonus  = 0.80f;   // Max extra multiplier from goal proximity
    float goalWidthGate      = 1500.f;  // Opponent must be within this X-width of goal

    // Ball-toward-goal velocity: bumping while ball is threatening is valuable
    float ballVelGoalBonus   = 0.40f;   // Max extra multiplier from ball going toward goal

    // ====================================================================
    // DRIBBLE-BUMP COMBO — the playstyle you want
    // ====================================================================
    float dribbleMaxDist     = 300.f;   // Max ball distance to count as dribbling
    float dribbleBumpBonus   = 1.50f;   // Extra multiplier when bumping WHILE dribbling
    float dribbleDemoBonus   = 2.00f;   // Extra multiplier when demoing WHILE dribbling

    // ====================================================================
    // DEMOED PENALTY — survival instinct
    // ====================================================================
    float demoedPenalty      = 0.25f;   // Penalty magnitude for getting demoed

    // ====================================================================
    // IMPLEMENTATION
    // ====================================================================

    virtual void Reset(const GameState& initialState) override {}

    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float reward = 0.f;

        // ------------------------------------------------------------------
        // PENALTY: getting demoed yourself
        // ------------------------------------------------------------------
        if (player.eventState.demoed) {
            reward -= demoedPenalty;
        }

        if (player.isDemoed)
            return reward;

        // ------------------------------------------------------------------
        // TEAM-RELATIVE GOALS (fixes the Blue-only bug)
        // ------------------------------------------------------------------
        Vec oppGoalCenter = (player.team == Team::BLUE)
            ? CommonValues::ORANGE_GOAL_CENTER
            : CommonValues::BLUE_GOAL_CENTER;
        Vec oppGoalMouth = (player.team == Team::BLUE)
            ? Vec(0.f, CommonValues::BACK_WALL_Y, CommonValues::GOAL_HEIGHT / 2.f)
            : Vec(0.f, -CommonValues::BACK_WALL_Y, CommonValues::GOAL_HEIGHT / 2.f);

        // ------------------------------------------------------------------
        // BALL THREAT: how fast is the ball going toward their goal?
        // ------------------------------------------------------------------
        Vec ballDirToGoal = (oppGoalMouth - state.ball.pos).Normalized();
        float ballVelTowardGoal = state.ball.vel.Dot(ballDirToGoal);
        float ballThreatFactor = RS_CLAMP(ballVelTowardGoal / 3000.f, 0.f, 1.f);

        // ------------------------------------------------------------------
        // ARE WE DRIBBLING? (ball close, similar velocity)
        // ------------------------------------------------------------------
        float distToBall = (state.ball.pos - player.pos).Length();
        bool isDribbling = distToBall < dribbleMaxDist;

        // ------------------------------------------------------------------
        // FIND BEST OPPONENT TARGET
        // ------------------------------------------------------------------
        float speed = player.vel.Length();
        Vec normVel = player.vel / RS_MAX(speed, 1.f);
        Vec forward = player.rotMat.forward;

        float bestApproach = 0.f;
        float bestTarget = 0.f;
        float bestClosing = 0.f;

        for (const auto& opp : state.players) {
            if (opp.team == player.team) continue;
            if (opp.carId == player.carId) continue;
            if (opp.isDemoed) continue;

            Vec toOpp = opp.pos - player.pos;
            float dist = toOpp.Length();
            if (dist > maxTargetDist || dist < 1.f) continue;

            Vec dirToOpp = toOpp / dist;

            // --- Context value of THIS opponent ---
            float ctx = _GetContext(opp, state.ball.pos, oppGoalCenter, ballThreatFactor);

            // --- Distance falloff: quadratic, ramps up when close ---
            float distFrac = 1.f - (dist / maxTargetDist);
            float distFactor = distFrac * distFrac;

            // --- APPROACH: velocity toward opponent while fast ---
            if (speed >= approachSpeedMin) {
                float velDot = normVel.Dot(dirToOpp);
                if (velDot > 0.f) {
                    float speedFrac = RS_MIN(1.f, speed / CommonValues::CAR_MAX_SPEED);
                    float val = velDot * speedFrac * distFactor * ctx;
                    bestApproach = RS_MAX(bestApproach, val);
                }
            }

            // --- TARGET: facing opponent while supersonic ---
            if (player.isSupersonic) {
                float faceDot = forward.Dot(dirToOpp);
                if (faceDot > 0.5f) {
                    float faceFrac = (faceDot - 0.5f) * 2.f; // remap 0.5→1.0 to 0→1
                    float val = faceFrac * distFactor * ctx;
                    bestTarget = RS_MAX(bestTarget, val);
                }
            }

            // --- CLOSING: distance shrinking fast ---
            if (player.prev) {
                float prevDist = (opp.pos - player.prev->pos).Length();
                float closingRate = prevDist - dist;
                if (closingRate > 0.f && state.deltaTime > 0.f) {
                    float closingFrac = RS_MIN(1.f, closingRate / (CommonValues::CAR_MAX_SPEED * state.deltaTime + 1.f));
                    float val = closingFrac * distFactor * ctx;
                    bestClosing = RS_MAX(bestClosing, val);
                }
            }
        }

        reward += approachWeight * bestApproach;
        reward += targetWeight * bestTarget;
        reward += closingWeight * bestClosing;

        // ------------------------------------------------------------------
        // EVENT: BUMP
        // ------------------------------------------------------------------
        if (player.eventState.bump && !player.eventState.demo) {
            float ctx = _GetBestVictimContext(player, state, oppGoalCenter, ballThreatFactor);
            float combo = isDribbling ? dribbleBumpBonus : 0.f;
            reward += bumpBase * (ctx + combo);
        }

        // ------------------------------------------------------------------
        // EVENT: DEMO
        // ------------------------------------------------------------------
        if (player.eventState.demo) {
            float ctx = _GetBestVictimContext(player, state, oppGoalCenter, ballThreatFactor);
            float combo = isDribbling ? dribbleDemoBonus : 0.f;
            reward += demoBase * (ctx + combo);
        }

        return reward;
    }

private:

    // Context value for a specific opponent (higher = more valuable target)
    // Returns a multiplier >= 1.0
    float _GetContext(const Player& opp, const Vec& ballPos,
                      const Vec& oppGoalCenter, float ballThreatFactor) {
        float bonus = 1.f; // base multiplier

        // --- Near the ball? ---
        float distToBall = (opp.pos - ballPos).Length();
        if (distToBall < ballProxRadius) {
            float frac = 1.f - (distToBall / ballProxRadius);
            bonus += ballProxMaxBonus * frac * frac;
        }

        // --- In front of their goal? ---
        float distToGoal = (opp.pos - oppGoalCenter).Length();
        bool withinGoalWidth = std::abs(opp.pos.x) < goalWidthGate;
        if (distToGoal < goalFrontRadius && withinGoalWidth) {
            float frac = 1.f - (distToGoal / goalFrontRadius);
            bonus += goalFrontMaxBonus * frac * frac;
        }

        // --- Ball threatening their goal? ---
        bonus += ballVelGoalBonus * ballThreatFactor;

        return bonus;
    }

    // Find the best context among opponents who got bumped/demoed this step
    // Falls back to nearest opponent if we can't identify the victim
    float _GetBestVictimContext(const Player& player, const GameState& state,
                                const Vec& oppGoalCenter, float ballThreatFactor) {

        // Try to find the actual victim (opponent who got bumped/demoed)
        float bestCtx = 0.f;
        bool foundVictim = false;

        for (const auto& opp : state.players) {
            if (opp.team == player.team) continue;

            bool isVictim = opp.eventState.bumped || opp.eventState.demoed;
            if (!isVictim) continue;

            // Sanity check: victim should be near us
            float dist = (opp.pos - player.pos).Length();
            if (dist > 600.f) continue; // Probably not our victim

            float ctx = _GetContext(opp, state.ball.pos, oppGoalCenter, ballThreatFactor);
            bestCtx = RS_MAX(bestCtx, ctx);
            foundVictim = true;
        }

        // Fallback: if we can't identify victim, use nearest opponent context
        if (!foundVictim) {
            float nearestDist = 99999.f;
            for (const auto& opp : state.players) {
                if (opp.team == player.team) continue;
                float dist = (opp.pos - player.pos).Length();
                if (dist < nearestDist) {
                    nearestDist = dist;
                    bestCtx = _GetContext(opp, state.ball.pos, oppGoalCenter, ballThreatFactor);
                }
            }
        }

        return bestCtx;
    }
};
	


    class DemoBumpNearBallReward : public Reward {
	public:
		// Radius within which a bump/demo gives reward (UU)
		float maxDist;

		// Reward scale for a plain bump
		float bumpScale;

		// Reward scale for a demo (applied on top of bumpScale since a demo
		// also counts as a bump in the event state)
		float demoScale;

		DemoBumpNearBallReward(
			float maxDist   = 1000.f,
			float bumpScale = 1.f,
			float demoScale = 3.f
		) : maxDist(maxDist),
		    bumpScale(bumpScale),
		    demoScale(demoScale) {
		}

		virtual void Reset(const GameState& state) override {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {

			bool didBump = player.eventState.bump;
			bool didDemo = player.eventState.demo;

			if (!didBump && !didDemo) return 0.f;

			// Find the victim - the nearest opponent that was bumped/demoed this step
			float    closestDist = FLT_MAX;
			for (const auto& other : state.players) {
				if (other.carId == player.carId) continue;  // Skip self
				if (other.team  == player.team)  continue;  // Skip teammates

				// Check if this opponent was on the receiving end
				if (!other.eventState.bumped && !other.eventState.demoed) continue;

				float d = (other.pos - state.ball.pos).Length();
				if (d < closestDist)
					closestDist = d;
			}

			if (closestDist == FLT_MAX) return 0.f; // No valid victim found

			// Proximity to ball: 1 at ball, 0 at maxDist.
			// Squared so reward drops off aggressively with distance —
			// a bump at half range gives 0.25x, not 0.5x.
			float proximity = 1.f - RS_CLAMP(closestDist / maxDist, 0.f, 1.f);
			proximity *= proximity; // square it
			if (proximity <= 0.f) return 0.f;

			float scale  = didDemo ? demoScale : bumpScale;
			float reward = scale * proximity;

			std::cout << "DemoBumpNearBallReward: "
			          << (didDemo ? "DEMO" : "bump")
			          << " dist=" << closestDist
			          << " proximity=" << proximity
			          << " reward=" << reward << std::endl;

			return reward;
		}
	};

	class StrategicDemoReward : public Reward {
	public:
		float baseReward;
		float minAlignment;      // below this alignment we treat as 0 bonus
		float maxDistanceForVictim; // sanity cap for victim distance

		StrategicDemoReward(
			float baseReward = 1.0f,
			float minAlignment = 0.2f,
			float maxDistanceForVictim = 8000.0f
		) : baseReward(baseReward),
			minAlignment(minAlignment),
			maxDistanceForVictim(maxDistanceForVictim) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// 1) Must be a demo event for this player (attacker)
			if (!player.eventState.demo) {
				return 0.0f;
			}

			// 2) Prefer the *actual* victim flagged by the env (demoed == true)
			const Player* victim = nullptr;
			for (const auto& p : state.players) {
				if (&p == &player) continue;
				if (p.eventState.demoed) {
					victim = &p;
					break;
				}
			}

			// 3) Fallback: if no explicit victim flagged, use closest opponent (robustness)
			if (!victim) {
				float bestDist = std::numeric_limits<float>::infinity();
				for (const auto& p : state.players) {
					if (p.team == player.team) continue;
					float d = (p.pos - player.pos).Length();
					if (d < bestDist) {
						bestDist = d;
						victim = &p;
					}
				}
			}

			if (!victim) {
				// no opponent found (weird), give base reward
				return baseReward;
			}

			// 4) Sanity: if victim extremely far (shouldn't happen), cap or fallback
			float distToVictim = (victim->pos - player.pos).Length();
			if (distToVictim > maxDistanceForVictim) {
				// suspicious — fall back to baseReward
				return baseReward;
			}

			// 5) Compute alignment: dot(normalized velocity, normalized direction to victim)
			Vec toVictim = victim->pos - player.pos;
			float toVictimLen = toVictim.Length();
			if (toVictimLen < 1e-3f) toVictimLen = 1e-3f;
			Vec dirToVictim = toVictim / toVictimLen;

			float speed = player.vel.Length();
			// If stationary, we consider alignment 0 (no directional intent)
			if (speed < 1e-3f) speed = 0.0f;

			float alignment = 0.0f;
			if (speed > 0.0f) {
				Vec velNorm = player.vel / speed;
				alignment = velNorm.Dot(dirToVictim); // -1 .. +1
			}
			else {
				alignment = 0.0f;
			}

			// 6) Only positive forward alignment above threshold gives bonus
			float bonusFactor = 0.0f;
			if (alignment > minAlignment) {
				// map alignment (minAlignment..1) -> (0..1)
				float norm = (alignment - minAlignment) / (1.0f - minAlignment);
				bonusFactor = std::clamp(norm, 0.0f, 1.0f);
			}

			// 7) Form final reward (base + scaled bonus). Tweak multiplier here.
			float finalReward = baseReward * (1.0f + bonusFactor);

			// DEBUG (useful during testing)
			std::cout << "[StrategicDemoReward] playerIdx=" << player.index
				<< " demoed someone. VictimCarId=" << victim->carId
				<< " dist=" << distToVictim
				<< " alignment=" << alignment
				<< " bonusFactor=" << bonusFactor
				<< " finalReward=" << finalReward << std::endl;

			return finalReward;
		}
	};

	class MawkzyFlickReward : public Reward {
	private:
		struct PlayerState {
			bool ballControlPhase = false;
			bool airRollPhase = false;
			bool flickPhase = false;
			bool flickCompleted = false;
			Vec ballPosition = Vec(0, 0, 0);
			Vec playerForward = Vec(0, 0, 0);
			Vec playerRight = Vec(0, 0, 0);
			float airRollTime = 0.0f;
			float flickPower = 0.0f;
			float airRollDirection = 0.0f; // -1 = gauche, +1 = droite
			bool wasOnGround = false;
			bool hadFlipBefore = false;
			Vec lastBallVelocity = Vec(0, 0, 0);
		};
		std::map<uint32_t, PlayerState> playerStates;

	public:
		float ballProximityThreshold;
		float minAirRollTime;
		float minFlickPower;
		float targetFlickAngle;
		float angleTolerance;
		float ballControlReward;
		float airRollReward;
		float flickReward;
		float powerBonus;
		float precisionBonus;
		float angleBonus;
		float directionBonus;
		bool debug;

		MawkzyFlickReward(
			float ballProximityThreshold = 250.0f,
			float minAirRollTime = 0.2f,
			float minFlickPower = 800.0f,
			float targetFlickAngle = 45.0f,
			float angleTolerance = 15.0f,
			float ballControlReward = 1.0f,
			float airRollReward = 2.0f,
			float flickReward = 4.0f,
			float powerBonus = 2.0f,
			float precisionBonus = 2.0f,
			float angleBonus = 3.0f,
			float directionBonus = 2.5f,
			bool debug = false
		) : ballProximityThreshold(ballProximityThreshold), minAirRollTime(minAirRollTime),
			minFlickPower(minFlickPower), targetFlickAngle(targetFlickAngle),
			angleTolerance(angleTolerance), ballControlReward(ballControlReward),
			airRollReward(airRollReward), flickReward(flickReward),
			powerBonus(powerBonus), precisionBonus(precisionBonus),
			angleBonus(angleBonus), directionBonus(directionBonus), debug(debug) {
		}

		virtual void Reset(const GameState& initialState) override {
			playerStates.clear();
			for (const auto& player : initialState.players) {
				PlayerState& state = playerStates[player.carId];
				state.ballControlPhase = false;
				state.airRollPhase = false;
				state.flickPhase = false;
				state.flickCompleted = false;
				state.ballPosition = Vec(0, 0, 0);
				state.playerForward = player.rotMat.forward;
				state.playerRight = player.rotMat.right;
				state.airRollTime = 0.0f;
				state.flickPower = 0.0f;
				state.airRollDirection = 0.0f;
				state.wasOnGround = player.isOnGround;
				state.hadFlipBefore = player.HasFlipOrJump();
				state.lastBallVelocity = Vec(0, 0, 0);
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0.0f;

			int carId = player.carId;
			if (playerStates.find(carId) == playerStates.end()) {
				PlayerState& newState = playerStates[player.carId];
				newState.ballControlPhase = false;
				newState.airRollPhase = false;
				newState.flickPhase = false;
				newState.flickCompleted = false;
				newState.ballPosition = Vec(0, 0, 0);
				newState.playerForward = player.rotMat.forward;
				newState.playerRight = player.rotMat.right;
				newState.airRollTime = 0.0f;
				newState.flickPower = 0.0f;
				newState.airRollDirection = 0.0f;
				newState.wasOnGround = player.isOnGround;
				newState.hadFlipBefore = player.HasFlipOrJump();
				newState.lastBallVelocity = Vec(0, 0, 0);
			}

			PlayerState& st = playerStates[carId];
			float reward = 0.0f;

			Vec playerPos = player.pos;
			Vec ballPos = state.ball.pos;
			float distanceToBall = (playerPos - ballPos).Length();
			bool isOnGround = player.isOnGround;
			Vec currentForward = player.rotMat.forward;
			Vec currentRight = player.rotMat.right;

			// === PHASE 1: CONTRÔLE DE LA BALLE ===
			if (distanceToBall < ballProximityThreshold && !st.ballControlPhase) {
				st.ballControlPhase = true;
				reward += ballControlReward;

				if (debug) {
					printf("[MawkzyFlick] car_id=%d PHASE 1: Contrôle de la balle! Reward: %.2f\n",
						carId, ballControlReward);
				}
			}

			// === PHASE 2: AIR ROLL DIRECTIONNEL ===
			if (st.ballControlPhase && !isOnGround && !st.airRollPhase) {
				// Détecter l'air roll via le changement de rotation
				if (st.playerForward.Length() > 0) {
					float rotationChange = fabsf(currentForward.Dot(st.playerForward));

					if (rotationChange < 0.8f) { // Rotation significative
						st.airRollPhase = true;

						// DÉTECTER LA DIRECTION DE L'AIR ROLL
						Vec carRight = player.rotMat.right;
						float airRollDirection = carRight.z; // -1 = gauche, +1 = droite

						// Normaliser la direction
						if (abs(airRollDirection) > 0.1f) {
							st.airRollDirection = (airRollDirection > 0) ? 1.0f : -1.0f;

							if (debug) {
								printf("[MawkzyFlick] car_id=%d PHASE 2: Air roll %s détecté! Direction: %.1f\n",
									carId, (st.airRollDirection > 0) ? "DROITE" : "GAUCHE", st.airRollDirection);
							}
						}

						reward += airRollReward;
					}
				}
			}

			// === PHASE 3: FLICK 45° ===
			if (st.airRollPhase && !st.flickPhase) {
				bool hasFlipNow = player.HasFlipOrJump();

				if (st.hadFlipBefore && !hasFlipNow) {
					st.flickPhase = true;
					st.lastBallVelocity = state.prev ? state.prev->ball.vel : Vec(0, 0, 0);

					if (debug) {
						printf("[MawkzyFlick] car_id=%d PHASE 3: Flick déclenché! Direction air roll: %.1f\n",
							carId, st.airRollDirection);
					}
				}
			}

			// === PHASE 4: VÉRIFICATION COMPLÈTE ===
			if (st.flickPhase && !st.flickCompleted && player.ballTouchedStep) {
				// 1. VÉRIFIER LA PUISSANCE DU FLICK
				float ballSpeedChange = 0.0f;
				if (state.prev) {
					ballSpeedChange = (state.ball.vel - st.lastBallVelocity).Length();
					st.flickPower = ballSpeedChange;
				}

				if (ballSpeedChange > minFlickPower) {
					reward += flickReward;
					reward += (ballSpeedChange - minFlickPower) * powerBonus / 1000.0f;

					if (debug) {
						printf("[MawkzyFlick] car_id=%d Puissance flick: %.0f, Reward: %.2f\n",
							carId, ballSpeedChange, reward);
					}
				}

				// 2. VÉRIFIER L'ANGLE DU FLICK (45°)
				float flickAngle = CalculateFlickAngle(player, state);
				float angleAccuracy = CalculateAngleAccuracy(flickAngle);

				if (angleAccuracy > 0.7f) { // Angle proche de 45°
					reward += angleBonus * angleAccuracy;

					if (debug) {
						printf("[MawkzyFlick] car_id=%d Angle flick: %.1f° (cible: 45°), Accuracy: %.2f, Bonus: %.2f\n",
							carId, flickAngle, angleAccuracy, angleBonus * angleAccuracy);
					}
				}

				// 3. VÉRIFIER LA DIRECTION DU FLICK (correspond à l'air roll)
				bool directionMatches = CheckDirectionMatch(st.airRollDirection, state.ball.vel);

				if (directionMatches) {
					reward += directionBonus;

					if (debug) {
						printf("[MawkzyFlick] car_id=%d DIRECTION PARFAITE! Air roll %s ? Flick %s, Bonus: %.2f\n",
							carId,
							(st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
							(st.airRollDirection > 0) ? "DROITE" : "GAUCHE",
							directionBonus);
					}
				}

				// 4. VÉRIFIER LA PRÉCISION VERS LE BUT
				Vec goalDirection = GetGoalDirection(player.team);
				Vec ballDirection = state.ball.vel.Normalized();
				float precision = ballDirection.Dot(goalDirection);

				if (precision > 0.6f) { // Précision vers le but
					reward += precisionBonus * precision;

					if (debug) {
						printf("[MawkzyFlick] car_id=%d Précision vers le but: %.2f, Bonus: %.2f\n",
							carId, precision, precisionBonus * precision);
					}
				}

				// 5. BONUS FINAL POUR MAWKZY FLICK PARFAIT
				if (angleAccuracy > 0.8f && directionMatches && precision > 0.7f) {
					float perfectBonus = 5.0f;
					reward += perfectBonus;

					if (debug) {
						printf("[MawkzyFlick] car_id=%d ?? MAWKZY FLICK PARFAIT! Bonus final: %.2f\n",
							carId, perfectBonus);
					}
				}

				st.flickCompleted = true;
			}

			// Récompense continue pour maintenir l'air roll
			if (st.airRollPhase && !isOnGround) {
				st.airRollTime += 1.0f / 120.0f;
				if (st.airRollTime > minAirRollTime) {
					reward += 0.1f; // Récompense continue
				}
			}

			// Reset si le joueur touche le sol
			if (st.wasOnGround && !isOnGround) {
				st.airRollTime = 0.0f;
			}

			// Mise à jour de l'état
			st.ballPosition = ballPos;
			st.playerForward = currentForward;
			st.playerRight = currentRight;
			st.wasOnGround = isOnGround;
			st.hadFlipBefore = player.HasFlipOrJump();

			return reward;
		}

	private:
		// Calculer l'angle du flick (0° = droit, 90° = vertical)
		float CalculateFlickAngle(const Player& player, const GameState& state) const {
			Vec carForward = Vec(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
			Vec ballVelocity = Vec(state.ball.vel.x, state.ball.vel.y, 0.0f);

			if (carForward.Length() < 0.1f || ballVelocity.Length() < 100.0f) {
				return 0.0f;
			}

			carForward = carForward.Normalized();
			ballVelocity = ballVelocity.Normalized();

			float dotProduct = RS_MAX(-1.0f, RS_MIN(1.0f, carForward.Dot(ballVelocity)));
			float angle = acosf(dotProduct);
			return angle * 180.0f / M_PI;
		}

		// Calculer la précision de l'angle (0-1, 1 = parfait 45°)
		float CalculateAngleAccuracy(float flickAngle) const {
			float angleDiff = abs(flickAngle - targetFlickAngle);
			if (angleDiff <= angleTolerance) {
				return 1.0f - (angleDiff / angleTolerance);
			}
			return 0.0f;
		}

		// Vérifier que la direction du flick correspond à l'air roll
		bool CheckDirectionMatch(float airRollDirection, const Vec& ballVelocity) const {
			if (abs(airRollDirection) < 0.1f) return false;

			// Air roll DROITE (+1) ? Flick doit aller vers la DROITE (X positif)
			// Air roll GAUCHE (-1) ? Flick doit aller vers la GAUCHE (X négatif)
			if (airRollDirection > 0) { // Droite
				return ballVelocity.x > 200.0f; // Vitesse X positive
			}
			else { // Gauche
				return ballVelocity.x < -200.0f; // Vitesse X négative
			}
		}

		Vec GetGoalDirection(Team team) {
			if (team == Team::BLUE) {
				return (CommonValues::ORANGE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
			}
			else {
				return (CommonValues::BLUE_GOAL_CENTER - Vec(0, 0, 0)).Normalized();
			}
		}
	};






	class BumpChainReward : public Reward {
	public:
		float baseBumpReward;
		int chainWindowTicks; // Use TICKS (frames), not seconds
		float maxChainMultiplier;

		// Internal State
		int currentChain;
		long long lastBumpTick; // Stores the tick number of the last bump
		long long currentTick;  // Keeps track of time internally

		// 120 ticks is roughly 1 second in Rocket League (at 120hz)
		BumpChainReward(
			float baseBumpReward = 0.5f,
			int chainWindowTicks = 360, // ~3 seconds at 120hz
			float maxChainMultiplier = 5.0f
		) : baseBumpReward(baseBumpReward),
			chainWindowTicks(chainWindowTicks),
			maxChainMultiplier(maxChainMultiplier),
			currentChain(0),
			lastBumpTick(-99999),
			currentTick(0) {
		}

		virtual void Reset(const GameState& initialState) override {
			currentChain = 0;
			lastBumpTick = -99999;
			currentTick = 0; // Reset our internal clock
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Increment our internal clock every step
			currentTick++;

			if (!player.eventState.bump) {
				return 0.0f;
			}

			// Calculate how many ticks passed since last bump
			long long ticksSinceLast = currentTick - lastBumpTick;
			bool chainContinued = false;

			// Logic: Did we bump fast enough?
			if (ticksSinceLast <= chainWindowTicks) {
				currentChain++;
				chainContinued = true;
			}
			else {
				currentChain = 1; // Start new chain
			}

			// Update last bump time
			lastBumpTick = currentTick;

			// Calculate Reward
			float multiplier = std::min((float)currentChain, maxChainMultiplier);
			float finalReward = baseBumpReward * multiplier;

			// --- DEBUG PRINT ---
			std::cout << "\n[BUMP CHAIN] >>> CONTACT DETECTED <<<" << std::endl;
			if (chainContinued) {
				std::cout << "  > Status:      CHAIN CONTINUED! (" << ticksSinceLast << " ticks since last)" << std::endl;
			}
			else {
				std::cout << "  > Status:      NEW CHAIN STARTED" << std::endl;
			}
			std::cout << "  > Chain Count: " << currentChain << " (Multiplier: x" << multiplier << ")" << std::endl;
			std::cout << "  > REWARD:      " << finalReward << std::endl;
			std::cout << "-------------------------------------\n" << std::endl;

			return finalReward;
		}
	};



	class HalfFlipReward : public Reward {
	private:
		float reward_value;
		float min_speed_gain;
		float min_facing_before;
		float min_facing_after;
		int min_ticks;
		bool debug;

		struct PlayerState {
			bool backflip_started = false;
			bool backflip_canceled = false;
			bool air_roll_detected = false;
			bool halfflip_completed = false;
			bool rewarded = false;
			float pre_flip_vel = 0.0f;
			float post_flip_vel = 0.0f;
			int flip_tick = 0;
			float facing_before = 0.0f;
			Vec initial_rotation = Vec(0, 0, 0);
			Vec rotation_at_cancel = Vec(0, 0, 0);
			Vec final_rotation = Vec(0, 0, 0);
		};

		std::unordered_map<int, PlayerState> player_states;

	public:
		HalfFlipReward(float reward_value = 8.0f,
			float min_speed_gain = 300.0f,
			float min_facing_before = -0.5f,
			float min_facing_after = 0.5f,
			int min_ticks = 4,
			bool debug = false)
			: reward_value(reward_value),
			min_speed_gain(min_speed_gain),
			min_facing_before(min_facing_before),
			min_facing_after(min_facing_after),
			min_ticks(min_ticks),
			debug(debug) {
		}

		virtual void Reset(const GameState& initial_state) override {
			player_states.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			int car_id = player.carId;
			if (player_states.find(car_id) == player_states.end()) {
				player_states[car_id] = PlayerState{};
			}

			PlayerState& st = player_states[car_id];
			float reward = 0.0f;

			// Direction vers la balle
			Vec ball_dir = state.ball.pos - player.pos;
			ball_dir = ball_dir.Normalized();

			// Direction de la voiture
			Vec forward = player.rotMat.forward;
			float facing_ball = forward.Dot(ball_dir);

			// Détection du début du backflip (saut + rotation vers l'arrière)
			if (!st.backflip_started && player.isJumping && !player.isOnGround) {
				// Vérifier si la voiture commence à se retourner (rotation pitch négative)
				Vec car_up = player.rotMat.up;
				if (car_up.z < 0.3f) { // La voiture commence à se retourner
					st.backflip_started = true;
					st.flip_tick = 0;
					st.pre_flip_vel = -forward.Dot(player.vel);
					st.facing_before = facing_ball;
					st.initial_rotation = Vec(car_up.x, car_up.y, car_up.z);

				}
			}

			// Détection du cancel du backflip
			if (st.backflip_started && !st.backflip_canceled) {
				st.flip_tick++;

				// Le cancel se caractérise par une rotation qui s'arrête
				Vec car_up = player.rotMat.up;
				float rotation_change = fabsf(car_up.z - st.initial_rotation.z);

				if (st.flip_tick >= min_ticks && rotation_change < 0.1f) {
					st.backflip_canceled = true;
					st.rotation_at_cancel = Vec(car_up.x, car_up.y, car_up.z);

				}
			}

			// Détection de l'air roll pour se retourner
			if (st.backflip_canceled && !st.air_roll_detected) {
				Vec car_up = player.rotMat.up;
				Vec car_forward = player.rotMat.forward;

				// L'air roll se caractérise par une rotation continue autour de l'axe forward
				float rotation_progress = fabsf(car_up.z - st.rotation_at_cancel.z);

				if (rotation_progress > 0.3f) { // Rotation significative détectée
					st.air_roll_detected = true;

				}
			}

			// Vérification du halfflip complet
			if (st.backflip_canceled && st.air_roll_detected && !st.halfflip_completed) {
				// Vérifier que la voiture fait face à la balle maintenant
				if (facing_ball > min_facing_after) {
					st.halfflip_completed = true;
					st.post_flip_vel = forward.Dot(player.vel);
					float speed_gain = st.post_flip_vel + st.pre_flip_vel;

					if (speed_gain > min_speed_gain) {
						reward = reward_value;

					}
				}
			}

			// Reset si au sol ou trop longtemps après le flip
			if (player.isOnGround || st.flip_tick > 50) {
				st.backflip_started = false;
				st.backflip_canceled = false;
				st.air_roll_detected = false;
				st.halfflip_completed = false;
				st.rewarded = false;
				st.flip_tick = 0;
			}

			return reward;
		}
	};

	class AirDribbleReward : public Reward {
	public:
		float minHeight;
		float maxHeight;
		float distanceThreshold;

		AirDribbleReward(float minHeight = 350.0f, float maxHeight = 1500.0f, float distanceThreshold = 400.0f)
			: minHeight(minHeight), maxHeight(maxHeight), distanceThreshold(distanceThreshold) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround)
				return 0.0f;

			if (state.ball.pos.z < minHeight)
				return 0.0f;

			float dist = (state.ball.pos - player.pos).Length();
			if (dist > distanceThreshold)
				return 0.0f;

			bool targetOrangeGoal = player.team == Team::BLUE;
			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec toGoal = (targetPos - state.ball.pos).Normalized();

			float velProj = state.ball.vel.Dot(toGoal);

			if (velProj <= 0.0f)
				return 0.0f;

			float velReward = velProj / CommonValues::BALL_MAX_SPEED;

			float clampedHeight = std::min(state.ball.pos.z, maxHeight);
			float heightScale = clampedHeight / maxHeight;

			return velReward * heightScale;
		}
	};

	class ShadowDefenseReward : public Reward {
	public:
		float maxBallGoalDistance;
		float maxLateralOffset;
		float idealSpacingFromBall;
		float spacingTolerance;
		ShadowDefenseReward(
			float maxBallGoalDistance = 3500.f,
			float maxLateralOffset = 1200.f,
			float idealSpacingFromBall = 1200.f,
			float spacingTolerance = 700.f
		) : maxBallGoalDistance(maxBallGoalDistance),
			maxLateralOffset(maxLateralOffset),
			idealSpacingFromBall(idealSpacingFromBall),
			spacingTolerance(spacingTolerance) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec goalCenter = player.team == Team::BLUE ? CommonValues::BLUE_GOAL_CENTER : CommonValues::ORANGE_GOAL_CENTER;
			float goalDirection = player.team == Team::BLUE ? -1.f : 1.f;

			Vec goalToBall = state.ball.pos - goalCenter;
			if (goalDirection * goalToBall.y < 0.f)
				return 0.f;

			float ballDist = goalToBall.Length();
			if (ballDist > maxBallGoalDistance || ballDist < 100.f)
				return 0.f;

			Vec goalDir = goalToBall / ballDist;

			Vec goalToPlayer = player.pos - goalCenter;
			float playerDepth = goalDir.Dot(goalToPlayer);
			if (playerDepth < 0.f || playerDepth > ballDist + 250.f)
				return 0.f;

			Vec lateral = goalToPlayer - goalDir * playerDepth;
			float lateralOffset = lateral.Length();
			float lateralWeight = 1.f - RS_CLAMP(lateralOffset / maxLateralOffset, 0.f, 1.f);
			if (lateralWeight <= 0.f)
				return 0.f;

			float spacing = ballDist - playerDepth;
			float spacingWeight = 1.f - RS_CLAMP(fabsf(spacing - idealSpacingFromBall) / spacingTolerance, 0.f, 1.f);
			if (spacingWeight <= 0.f)
				return 0.f;

			Vec toBall = state.ball.pos - player.pos;
			float facingWeight = 0.f;
			if (toBall.Length() > 100.f)
				facingWeight = RS_MAX(0.f, player.rotMat.forward.Dot(toBall.Normalized()));

			float velocityWeight = 0.f;
			if (player.vel.Length() > 100.f) {
				Vec desired = toBall.Normalized();
				Vec defensiveDir = Vec(0.f, goalDirection, 0.f);
				Vec blendBase = desired * 0.6f + defensiveDir * 0.4f;
				if (blendBase.Length() < 1e-3f)
					blendBase = desired;
				Vec blended = blendBase.Normalized();
				velocityWeight = RS_MAX(0.f, player.vel.Normalized().Dot(blended));
			}

			float controlWeight = 0.4f * facingWeight + 0.6f * velocityWeight;
			return lateralWeight * spacingWeight * controlWeight;
		}
	};









	class ControlReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.ballTouchedStep || !player.prev) {
				return 0;
			}

			// Calculate distance to ball
			float dist = player.pos.Dist(state.ball.pos);

			// Reward is higher the closer the player is to the ball, max reward is 1.
			// The max distance is set based on a rough estimate of dribble distance.
			constexpr float MAX_DIST = 500.f;
			return exp(-0.5 * (dist / MAX_DIST) * (dist / MAX_DIST));
		}
	};

	class FastDribblePopBumpReward : public Reward {
	private:
		struct PlayerState {
			bool isDribbling = false;
			int dribbleTouches = 0;
			int ticksSinceLastTouch = 0;
			bool popped = false;
			int ticksSincePop = 0;
			bool canChase = false;
			int ticksSinceChase = 0;
		};

		std::map<uint32_t, PlayerState> st;

	public:
		virtual void Reset(const GameState& s) override {
			st.clear();
		}

		virtual float GetReward(const Player& p, const GameState& s, bool f) override {
			if (!s.prev) return 0.f;
			auto& ps = st[p.carId];

			ps.ticksSinceLastTouch++;
			ps.ticksSincePop++;
			ps.ticksSinceChase++;

			float r = 0.f;

			float dist = (p.pos - s.ball.pos).Length();
			float spd = p.vel.Length();
			float spdNorm = spd / CAR_MAX_SPEED;

			bool drib = p.isOnGround &&
				s.ball.pos.z < 250.f &&
				dist < 180.f &&
				spd > 300.f;

			if (drib) {
				r += 0.15f * spdNorm;

				if (p.ballTouchedStep) {
					if (ps.ticksSinceLastTouch < 50) ps.dribbleTouches++;
					else ps.dribbleTouches = 1;
					ps.ticksSinceLastTouch = 0;
				}
			}

			ps.isDribbling = drib;

			bool up = s.ball.vel.z > 300.f;
			bool rising = s.ball.pos.z > 280.f && s.prev->ball.pos.z <= 280.f;
			bool hasTouches = ps.dribbleTouches >= 2;

			if (drib && hasTouches && rising && up && p.ballTouchedStep) {
				r += 3.f;
				ps.popped = true;
				ps.ticksSincePop = 0;
				ps.canChase = true;
				ps.ticksSinceChase = 0;
			}

			if (ps.ticksSincePop > 6) ps.popped = false;
			if (ps.ticksSinceChase > 90) ps.canChase = false;

			const Player* opp = nullptr;
			float dmin = 1e9f;
			for (auto& o : s.players) {
				if (o.team != p.team) {
					float d = (o.pos - p.pos).Length();
					if (d < dmin) { dmin = d; opp = &o; }
				}
			}
			if (!opp) return r;

			Vec toOpp = (opp->pos - p.pos).Normalized();
			float faceOpp = p.rotMat.forward.Dot(toOpp);
			float towardsOpp = p.vel.Normalized().Dot(toOpp);

			if (drib) r += 0.15f * faceOpp;

			if (ps.canChase) {
				r += 0.25f * faceOpp;
				r += 0.3f * towardsOpp;
				r += 0.2f * spdNorm;

				if (spd < 900.f) r -= 0.15f;
				if (spd > SUPERSONIC_THRESHOLD) r += 0.3f;
			}

			if (ps.canChase && (p.eventState.bump || p.eventState.demo)) {
				if (p.eventState.demo) r += 25.f;
				else r += 12.f;
				r += 5.f;
				ps.canChase = false;
				ps.dribbleTouches = 0;
			}

			return r;
		}
	};

	class DemoChaseReward : public Reward {
	public:
		const float BALL_RADIUS = 92.75f;
		const float TICKS_PER_SECOND = 120.f;

		const float CHASE_REWARD_SCALE = 10.0f;  // Scale for chasing an opponent
		const float SPEED_BONUS_SCALE = 2.0f;    // Scale for speed when chasing
		const float DEMO_REWARD = 30.0f;         // Large reward for a successful demo
		const float CHASE_DISTANCE_THRESHOLD = 1500.0f;  // Distance threshold to start chasing

	public:
		DemoChaseReward() {}

		virtual void Reset(const GameState& initialState) override {
			// Reset reward state (no specific state to reset for now)
		}

		// Function to find the closest opponent
		Player FindClosestOpponent(const Player& player, const GameState& state) {
			float minDistance = FLT_MAX;
			Player closestOpponent;

			for (const Player& opponent : state.players) {
				if (opponent.team == player.team || opponent.carId == player.carId) {
					continue; // Skip self or teammates
				}
				float distance = (player.pos - opponent.pos).Length();
				if (distance < minDistance) {
					minDistance = distance;
					closestOpponent = opponent;
				}
			}

			return closestOpponent;
		}

		// Function to calculate the chase reward based on distance and velocity
		float GetChaseReward(const Player& player, const GameState& state, const Player& opponent) {
			Vec dirToOpponent = opponent.pos - player.pos;
			float distanceToOpponent = dirToOpponent.Length();

			// Normalize the direction to the opponent
			if (distanceToOpponent > 0) {
				dirToOpponent = dirToOpponent / distanceToOpponent;  // Normalize the vector
			}

			float chaseReward = 0.0f;
			if (distanceToOpponent < CHASE_DISTANCE_THRESHOLD) {
				// Reward for being close and moving toward the opponent
				float alignment = dirToOpponent.Dot(player.vel.Normalized());
				chaseReward = CHASE_REWARD_SCALE * (1.0f - (distanceToOpponent / CHASE_DISTANCE_THRESHOLD)) * alignment;
			}

			// Speed bonus: reward the player for moving faster towards the opponent
			float speedBonus = RS_CLAMP(player.vel.Length() / 1500.f, 0.5f, 1.5f);
			chaseReward += speedBonus * SPEED_BONUS_SCALE;

			return chaseReward;
		}

		// Function to check if the player demoed the opponent
		bool IsDemoingOpponent(const Player& player, const GameState& state) {
			for (const Player& opponent : state.players) {
				if (opponent.carId == player.carId) continue;  // Skip self
				Vec playerToOpponent = opponent.pos - player.pos;
				float distance = playerToOpponent.Length();
				if (distance < BALL_RADIUS + 20.0f && !opponent.isOnGround) {  // Demo condition
					return true;  // Demo detected
				}
			}
			return false;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !state.prev) return 0.0f;

			// Find the closest opponent
			Player closestOpponent = FindClosestOpponent(player, state);

			// Calculate the chase reward for pursuing the closest opponent
			float chaseReward = GetChaseReward(player, state, closestOpponent);

			// Add demo reward if player demoed an opponent
			if (IsDemoingOpponent(player, state)) {
				chaseReward += DEMO_REWARD;
			}

			return chaseReward;
		}
	};


	class FlipResetReward : public Reward {
	private:
		struct PlayerState {
			bool lastHasFlip;
			bool resetDetected;
			bool setupActive;
			float lastBallDistance;
			float lastRollAngle;
			float lastAirTime;
			int resetFrame;

			PlayerState()
				: lastHasFlip(false)
				, resetDetected(false)
				, setupActive(false)
				, lastBallDistance(std::numeric_limits<float>::infinity())
				, lastRollAngle(0.0f)
				, lastAirTime(0.0f)
				, resetFrame(0)
			{
			}
		};

		float resetReward;
		float setupBonus;
		float proximityBonus;
		float invertedBonus;
		float weight;
		float minAirHeight;
		float maxBallDistance;
		float invertedThreshold;
		bool debug;

		std::unordered_map<uint32_t, PlayerState> playerStates;

		bool IsInAir(const Player& player) const {
			return player.pos.z > minAirHeight;
		}

		bool IsCarInverted(const Player& player) const {
			Angle angle = Angle::FromRotMat(player.rotMat);
			float rollAngleDeg = std::abs(angle.roll * 180.0f / M_PI);
			return std::abs(rollAngleDeg - 180.0f) < invertedThreshold;
		}

		bool IsCloseToBall(const Player& player, const GameState& state) const {
			float distance = (player.pos - state.ball.pos).Length();
			return distance <= maxBallDistance;
		}

		bool DetectWheelContact(const Player& player, const GameState& state) const {
			if (!IsCloseToBall(player, state)) {
				return false;
			}

			float distance = (player.pos - state.ball.pos).Length();
			return distance <= 100.0f && IsCarInverted(player);
		}

		bool DetectFlipReset(const Player& player, PlayerState& playerState) {
			bool currentHasFlip = player.HasFlipOrJump();
			bool lastHasFlip = playerState.lastHasFlip;

			bool flipResetDetected = !lastHasFlip && currentHasFlip;

			playerState.lastHasFlip = currentHasFlip;

			return flipResetDetected;
		}

		float CalculateSetupQuality(const Player& player, const GameState& state) const {
			float quality = 0.0f;

			if (IsInAir(player)) {
				quality += 0.3f;
			}

			if (IsCarInverted(player)) {
				quality += 0.3f;
			}

			float distance = (player.pos - state.ball.pos).Length();
			if (distance <= maxBallDistance) {
				float proximityScore = 1.0f - (distance / maxBallDistance);
				quality += 0.4f * proximityScore;
			}

			return quality;
		}

	public:
		FlipResetReward(
			float resetReward = 15.0f,
			float setupBonus = 2.0f,
			float proximityBonus = 1.0f,
			float invertedBonus = 1.0f,
			float weight = 1.0f,
			float minAirHeight = 50.0f,
			float maxBallDistance = 150.0f,
			float invertedThreshold = 150.0f,
			bool debug = false)
			: resetReward(resetReward)
			, setupBonus(setupBonus)
			, proximityBonus(proximityBonus)
			, invertedBonus(invertedBonus)
			, weight(weight)
			, minAirHeight(minAirHeight)
			, maxBallDistance(maxBallDistance)
			, invertedThreshold(invertedThreshold)
			, debug(debug)
		{
		}

		virtual void Reset(const GameState& initialState) override {
			playerStates.clear();
			for (const auto& player : initialState.players) {
				playerStates[player.carId] = PlayerState();
				playerStates[player.carId].lastHasFlip = player.HasFlipOrJump();
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (playerStates.find(player.carId) == playerStates.end()) {
				playerStates[player.carId] = PlayerState();
				playerStates[player.carId].lastHasFlip = player.HasFlipOrJump();
			}

			PlayerState& playerState = playerStates[player.carId];
			float reward = 0.0f;

			bool inAir = IsInAir(player);
			bool inverted = IsCarInverted(player);
			bool closeToBall = IsCloseToBall(player, state);
			bool wheelContact = DetectWheelContact(player, state);

			bool flipResetDetected = DetectFlipReset(player, playerState);

			if (inAir && inverted && closeToBall && wheelContact && flipResetDetected) {
				if (!playerState.resetDetected) {
					reward += resetReward;
					playerState.resetDetected = true;

					if (debug) {
						std::cout << "FLIP RESET DETECTED! Player " << player.carId
							<< " - Reward: " << resetReward << std::endl;
					}
				}
			}
			else if (inAir && inverted && closeToBall) {
				float setupQuality = CalculateSetupQuality(player, state);
				reward += setupBonus * setupQuality;

				if (!playerState.setupActive) {
					playerState.setupActive = true;
				}
			}
			else {
				if (playerState.setupActive) {
					playerState.setupActive = false;
				}
				if (playerState.resetDetected) {
					playerState.resetDetected = false;
				}
			}

			playerState.lastBallDistance = (player.pos - state.ball.pos).Length();

			Angle angle = Angle::FromRotMat(player.rotMat);
			playerState.lastRollAngle = std::abs(angle.roll * 180.0f / M_PI);

			return reward * weight;
		}
	};

	class KickoffFirstTouchReward : public Reward {
	private:
		bool _kickoff_active;
		int _first_touch_player_id_this_tick;

	public:
		KickoffFirstTouchReward() : _kickoff_active(false), _first_touch_player_id_this_tick(-1) {}
		void Reset(const GameState& initial_state) override {
			_kickoff_active = initial_state.ball.vel.LengthSq() < 1.f;
			_first_touch_player_id_this_tick = -1;
		}

		void PreStep(const GameState& state) override {
			_first_touch_player_id_this_tick = -1;

			if (!_kickoff_active) return;
			bool kickoff_ended = false;

			for (const auto& p : state.players) {
				if (p.ballTouchedStep) {
					_first_touch_player_id_this_tick = p.carId;
					kickoff_ended = true;
					break;
				}
			}

			if (!kickoff_ended && state.ball.vel.LengthSq() > 1.f) {
				kickoff_ended = true;
			}

			if (kickoff_ended) {
				_kickoff_active = false;
			}
		}

		class DribbleToAirReward : public Reward {
		public:
			const float BALL_RADIUS = 92.75f;
			const float TICKS_PER_SECOND = 120.f;

			const float DRIBBLE_REWARD_SCALE = 5.0f;
			const float MIN_DRIBBLE_TICKS_FOR_POP = 1.5f * TICKS_PER_SECOND; // 1.5 seconds
			//	const float PATIENCE_BONUS_SCALE = 0.0f;
				//const float BOOST_WASTE_LIMIT = 20.f;
				//const float BOOST_PENALTY_SCALE = -0.1f;

			const float POP_JACKPOT = 10.0f;
			const float GROUND_PENALTY = -20.0f;
			const float TILT_REWARD = 3.0f;
			const float BUMP_REWARD = 100.0f;  // Reward for bumping opponent
			const float DEMO_REWARD = 300.0f;  // Reward for demoing opponent

		private:
			std::map<int, int> agentState;
			std::map<int, int> agentDribbleTicks;
			std::map<int, float> agentBoostAtDribbleStart;

		public:
			DribbleToAirReward() {}

			virtual void Reset(const GameState& initialState) override {
				agentState.clear();
				agentDribbleTicks.clear();
				agentBoostAtDribbleStart.clear();
			}

			bool IsDribbling(const Player& player, const GameState& state) {
				if (!player.isOnGround) return false;
				Vec carToBall = state.ball.pos - player.pos;
				return (
					carToBall.Length() < 180.0f &&
					state.ball.pos.z > 100.0f &&
					state.ball.pos.z < 200.0f &&
					carToBall.Dot(player.rotMat.forward) > 50.0f
					);
			}

			float GetDribbleReward(const Player& player, const GameState& state, float boostStart) {
				Vec carToBall = state.ball.pos - player.pos;

				float lateralOffset = abs(carToBall.Dot(player.rotMat.right));
				float centeringReward = exp(-0.05f * lateralOffset);

				// Position Reward: Ideal is 95.0f (Stable, pop-ready)
				float forwardOffset = carToBall.Dot(player.rotMat.forward);
				float positionError = abs(forwardOffset - 95.f);
				float positionReward = exp(-0.04f * positionError);

				float speedReward = RS_CLAMP(player.vel.Length() / 1500.f, 0.5f, 1.0f);



				return (centeringReward * positionReward * speedReward);
			}

			bool IsBumpingOpponent(const Player& player, const GameState& state) {
				// Bump detection: check if the player is colliding with an opponent
				for (const Player& opponent : state.players) {
					if (opponent.carId == player.carId) continue; // Don't check self
					Vec playerToOpponent = opponent.pos - player.pos;
					float distance = playerToOpponent.Length();
					if (distance < BALL_RADIUS + 100.0f) {  // Somewhat close, assume a bump happened
						return true;
					}
				}
				return false;
			}

			bool IsDemoingOpponent(const Player& player, const GameState& state) {
				// Demo detection: check if the player demoed an opponent
				for (const Player& opponent : state.players) {
					if (opponent.carId == player.carId) continue; // Don't check self
					Vec playerToOpponent = opponent.pos - player.pos;
					float distance = playerToOpponent.Length();
					if (distance < BALL_RADIUS + 20.0f && !opponent.isOnGround) {  // Close and opponent is off ground
						return true;
					}
				}
				return false;
			}



			virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
				if (!player.prev || !state.prev) return 0.0;

				int agentId = player.carId;
				if (agentState.find(agentId) == agentState.end()) {
					agentState[agentId] = 0;
					agentDribbleTicks[agentId] = 0;
					agentBoostAtDribbleStart[agentId] = player.boost;
				}

				int currentState = agentState[agentId];
				float currentBoostStart = agentBoostAtDribbleStart[agentId];

				// STATE 0: ON GROUND
				if (currentState == 0) {
					if (IsDribbling(player, state)) {
						agentState[agentId] = 1;
						agentDribbleTicks[agentId] = 1;
						agentBoostAtDribbleStart[agentId] = player.boost;
						return GetDribbleReward(player, state, player.boost) * DRIBBLE_REWARD_SCALE;
					}
				}
				// STATE 1: DRIBBLING
				else if (currentState == 1) {
					agentDribbleTicks[agentId]++;

					bool justJumped = player.prev->isOnGround && !player.isOnGround;
					if (justJumped) {
						agentState[agentId] = 2;
						return 0.0f;
					}
					else if (!IsDribbling(player, state)) {
						agentState[agentId] = 0;
						agentDribbleTicks[agentId] = 0;
						return 0.0f;
					}
				}
				// STATE 2: AIRBORNE
				else if (currentState == 2) {
					if (player.isOnGround) {
						agentState[agentId] = 0;
						agentDribbleTicks[agentId] = 0;
						return GROUND_PENALTY;
					}
					if (player.hasFlipped) {
						agentState[agentId] = 0;
						agentDribbleTicks[agentId] = 0;
						return 0.0f;
					}
					if (state.ball.pos.z < 100) {
						agentState[agentId] = 0;
						agentDribbleTicks[agentId] = 0;
						return 0.0f;
					}

					bool justDoubleJumped = player.hasDoubleJumped && !player.prev->hasDoubleJumped;
					if (justDoubleJumped) {
						if (agentDribbleTicks[agentId] < MIN_DRIBBLE_TICKS_FOR_POP) {
							return -10.0f; // Punish early pop
						}

						float curVertVel = state.ball.vel.z;
						Vec ballHorizontalVel = Vec(state.ball.vel.x, state.ball.vel.y, 0);
						float horizontalSpeed = ballHorizontalVel.Length();

						float popAngle = atan2f(curVertVel, horizontalSpeed + 0.1f);
						const float idealAngle = 1.396f; // 80 degrees
						const float maxAngleError = 0.087f;

						float angleQuality = RS_MAX(0.f, 1.0f - (abs(popAngle - idealAngle) / maxAngleError));

						if (angleQuality < 0.2f) agentState[agentId] = 0;

						return POP_JACKPOT * angleQuality;
					}

					// Check the position of the player relative to the ball to give reward for going in front
					Vec dirToBall = (state.ball.pos - player.pos).Normalized();
					float forwardOffset = dirToBall.Dot(player.rotMat.forward); // How much player is in front of the ball
					float inFrontReward = RS_CLAMP(forwardOffset, 0.0f, 1.0f); // Reward the player for being in front

					// Reward based on speed while jumping and being in front of the ball
					float speedBonus = RS_CLAMP(player.vel.Length() / 1500.f, 0.5f, 1.5f); // Give more reward for higher speed
					float jumpSpeedReward = inFrontReward * speedBonus * 20.0f; // Strong reward for being in front and fast

					// Tilt reward (maintaining good control and positioning while airborne)
					float tiltReward = player.rotMat.forward.Dot(dirToBall);
					float totalReward = RS_MAX(0.f, tiltReward) * TILT_REWARD;

					// Add the jumping speed reward to the tilt reward
					totalReward += jumpSpeedReward;

					// Reward for bumping or demoing the opponent
					if (IsBumpingOpponent(player, state)) {
						totalReward += BUMP_REWARD;
					}

					if (IsDemoingOpponent(player, state)) {
						totalReward += DEMO_REWARD;
					}

					return totalReward;
				}

				return 0.0f;
			}
		};




		/**
		 * @brief Rewards scoring via a ceiling shot.
		 */
		class CeilingShotReward : public Reward {
		public:
			virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
				if (player.eventState.goal && player.prev) {
					// Crude check: was the player on the ceiling recently?
					// A proper implementation would need stateful tracking per player.
					const Player* p = player.prev;
					for (int i = 0; i < 120 * 3 && p != nullptr; ++i, p = p->prev) { // Check last 3 seconds
						if (p->worldContact.hasContact && p->worldContact.contactNormal.z < -0.9f) {
							return 1.0f;
						}
					}
				}
				return 0;
			}
		};

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (_first_touch_player_id_this_tick != -1) {
				if (player.carId == _first_touch_player_id_this_tick) {
					return 1.0f;
				}
				else {
					return -1.0f;
				}
			}

			return 0.0f;
		}
	};

	/**
	 * @brief Rewards the player for executing a 45-degree flick.
	 * NOTE: This is a simplified detection and rewards any flick that sends the ball upward.
	 * A more precise implementation would need to check car-to-ball orientation before the flick.
	 */
	class FlickReward45degree : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.ballTouchedStep || !player.prev || player.isOnGround) {
				return 0;
			}

			// Check if the player was on the ground and started flipping in the previous step
			if (player.prev->isOnGround && player.isFlipping) {
				float ball_vel_2d = state.ball.vel.Length2D();
				if (ball_vel_2d == 0) return 0;

				// Calculate the angle of the ball's velocity vector relative to the ground plane
				float angle = atan2(state.ball.vel.z, ball_vel_2d);

				// Reward for being close to a 45-degree angle
				float target_angle = M_PI / 4;
				float error = std::abs(angle - target_angle);

				// Use gaussian function to score similarity
				return exp(-0.5 * (error / (target_angle * 0.5)) * (error / (target_angle * 0.5)));
			}

			return 0;
		}
	};

	/**
	 * @brief Rewards the player for using air roll while airborne.
	 */
	class AirRollReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.isOnGround && player.prevAction.roll != 0) {
				return 1.0f;
			}
			return 0;
		}
	};

	/**
	 * @brief Rewards the player for moving towards the closest available boost pad.
	 */
	class BoostSeekingReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.boost >= 100) return 0; // Don't seek boost if full

			Vec closest_pad_pos;
			float closest_dist_sq = -1;

			auto& boost_pads = state.GetBoostPads(player.team == Team::ORANGE);
			for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; ++i) {
				if (boost_pads[i]) { // If boost pad is active
					Vec pad_pos = CommonValues::BOOST_LOCATIONS[i];
					float dist_sq = player.pos.DistSq(pad_pos);

					if (closest_dist_sq < 0 || dist_sq < closest_dist_sq) {
						closest_dist_sq = dist_sq;
						closest_pad_pos = pad_pos;
					}
				}
			}

			if (closest_dist_sq < 0) return 0; // No active boost pads

			Vec dir_to_boost = (closest_pad_pos - player.pos).Normalized();
			return player.vel.Dot(dir_to_boost) / CommonValues::CAR_MAX_SPEED;
		}
	};

	/**
	 * @brief Rewards the player for performing a wavedash.
	 * This is a more robust version that checks for the speed increase from the wavedash.
	 */
	class WaveDashReward2 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0;

			// Check for transition from flipping in air to being on ground
			if (player.isOnGround && player.prev->isFlipping && !player.prev->isOnGround) {
				// Reward the speed increase from the wavedash
				float speed_increase = player.vel.Length() - player.prev->vel.Length();
				if (speed_increase > 100) { // Must provide a meaningful speed boost
					return 1.0f;
				}
			}
			return 0;
		}
	};

	/**
	 * @brief Rewards the player for executing a half-flip.
	 * NOTE: A precise half-flip is hard to detect. This rewards quickly turning 180 degrees
	 * while on the ground, which is the primary outcome of a half-flip.
	 */
	class HalfFlipReward2 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.isOnGround || !player.prev || !player.prev->prev) return 0;

			// Check if the player was recently flipping backwards.
			if (player.prev->hasFlipped && player.prev->prevAction.pitch > 0.5) {
				// Check if the car is now facing the opposite direction.
				float dot = player.rotMat.forward.Dot(player.prev->prev->rotMat.forward);
				if (dot < -0.8) { // Facing nearly the opposite direction
					return 1.0f;
				}
			}
			return 0;
		}
	};

	/**
	 * @brief Rewards a speed-flip.
	 * NOTE: Precise detection is complex. This implementation rewards reaching supersonic
	 * very quickly from a near-standstill, a key outcome of a speed-flip.
	 */
	class AdvancedSpeedFlipReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || player.isSupersonic == player.prev->isSupersonic) return 0;

			// If we just became supersonic and our previous speed was low
			if (player.isSupersonic && player.prev->vel.Length() < 500) {
				return 1.0f;
			}

			return 0;
		}
	};

	/**
	 * @brief Rewards doing a speed-flip on kickoff.
	 * NOTE: This rewards getting to the ball quickly on kickoff as a proxy for a good kickoff.
	 */
	class KickoffSpeedFlipRewardV2 : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Check if it's a kickoff scenario (ball is at center)
			if (state.ball.pos.Length2D() < 10) {
				if (player.ballTouchedStep) {
					// Reward is inversely proportional to the time it took to touch the ball
					return 2.5f - state.ball.pos.z / (CommonValues::BALL_RADIUS * 2);
				}
			}
			return 0;
		}
	};



	class ProgressiveDribblePopBumpReward : public Reward {
	private:
		struct PlayerState {
			bool isDribbling = false;
			int dribbleTouches = 0;
			int ticksSinceLastDribbleTouch = 0;
			bool justPopped = false;
			int ticksSincePop = 0;
			bool eligibleForBumpReward = false;
			int ticksSinceEligible = 0;
		};

		std::map<uint32_t, PlayerState> states;

	public:
		float dribbleProgressReward;    // Continuous reward for dribbling
		float popReward;                // Reward for successful pop
		float bumpReward;               // Reward for bump after pop
		float demoReward;               // Reward for demo after pop
		float sequenceCompleteBonus;    // Extra bonus for full sequence

		ProgressiveDribblePopBumpReward(
			float dribbleProgressReward = 0.1f,
			float popReward = 5.0f,
			float bumpReward = 35.0f,
			float demoReward = 45.0f,
			float sequenceCompleteBonus = 10.0f
		) : dribbleProgressReward(dribbleProgressReward),
			popReward(popReward),
			bumpReward(bumpReward),
			demoReward(demoReward),
			sequenceCompleteBonus(sequenceCompleteBonus) {
		}

		virtual void Reset(const GameState& initialState) override {
			states.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev) return 0.0f;

			auto& pState = states[player.carId];

			pState.ticksSinceLastDribbleTouch++;
			pState.ticksSincePop++;
			pState.ticksSinceEligible++;

			float reward = 0.0f;

			// STEP 1: Reward dribbling progress
			float distToBall = (player.pos - state.ball.pos).Length();
			bool isDribblingNow = player.isOnGround &&
				state.ball.pos.z <= 250.0f &&
				distToBall <= 180.0f &&
				player.vel.Length() >= 300.0f;

			if (isDribblingNow) {
				reward += dribbleProgressReward;

				if (player.ballTouchedStep) {
					if (pState.ticksSinceLastDribbleTouch < 60) {
						pState.dribbleTouches++;
					}
					else {
						pState.dribbleTouches = 1;
					}
					pState.ticksSinceLastDribbleTouch = 0;
				}
			}



			pState.isDribbling = isDribblingNow;

			// STEP 2: Reward pop
			bool wasDribblingRecently = pState.ticksSinceLastDribbleTouch <= 20;
			bool hadEnoughTouches = pState.dribbleTouches >= 2;
			bool ballWentHigh = state.ball.pos.z >= 300.0f &&
				state.prev->ball.pos.z < 300.0f;
			bool ballMovingUp = state.ball.vel.z >= 400.0f;

			if (wasDribblingRecently && hadEnoughTouches && ballWentHigh && ballMovingUp && player.ballTouchedStep) {
				reward += popReward;
				pState.justPopped = true;
				pState.ticksSincePop = 0;
				pState.eligibleForBumpReward = true;
				pState.ticksSinceEligible = 0;
			}

			if (pState.ticksSincePop > 5) {
				pState.justPopped = false;
			}

			if (pState.ticksSinceEligible > 90) {
				pState.eligibleForBumpReward = false;
				pState.dribbleTouches = 0;
			}

			// STEP 3: Reward bump/demo after pop
			if (pState.eligibleForBumpReward &&
				(player.eventState.bump || player.eventState.demo)) {

				bool didDemo = player.eventState.demo;
				reward += didDemo ? demoReward : bumpReward;

				// Full sequence bonus!
				reward += sequenceCompleteBonus;

				pState.eligibleForBumpReward = false;
				pState.dribbleTouches = 0;
			}

			return reward;
		}
	};


	class DribbleToAirReward : public Reward {
	public:
		const float BALL_RADIUS = 92.75f;
		const float TICKS_PER_SECOND = 120.f;

		const float DRIBBLE_REWARD_SCALE = 5.0f;
		const float MIN_DRIBBLE_TICKS_FOR_POP = 1.5f * TICKS_PER_SECOND; // 1.5 seconds
		const float PATIENCE_BONUS_SCALE = 7.5f;
		const float BOOST_WASTE_LIMIT = 20.f;
		const float BOOST_PENALTY_SCALE = -0.1f;

		const float POP_JACKPOT = 10.0f;
		const float GROUND_PENALTY = -20.0f;
		const float TILT_REWARD = 3.0f;

	private:
		std::map<int, int> agentState;
		std::map<int, int> agentDribbleTicks;
		std::map<int, float> agentBoostAtDribbleStart;

	public:
		DribbleToAirReward() {}

		virtual void Reset(const GameState& initialState) override {
			agentState.clear();
			agentDribbleTicks.clear();
			agentBoostAtDribbleStart.clear();
		}

		bool IsDribbling(const Player& player, const GameState& state) {
			if (!player.isOnGround) return false;
			Vec carToBall = state.ball.pos - player.pos;
			return (
				carToBall.Length() < 180.0f &&
				state.ball.pos.z > 100.0f &&
				state.ball.pos.z < 200.0f &&
				carToBall.Dot(player.rotMat.forward) > 50.0f
				);
		}

		float GetDribbleReward(const Player& player, const GameState& state, float boostStart) {
			Vec carToBall = state.ball.pos - player.pos;

			float lateralOffset = abs(carToBall.Dot(player.rotMat.right));
			float centeringReward = exp(-0.05f * lateralOffset);

			// Position Reward: Ideal is 95.0f (Stable, pop-ready)
			float forwardOffset = carToBall.Dot(player.rotMat.forward);
			float positionError = abs(forwardOffset - 95.f);
			float positionReward = exp(-0.04f * positionError);

			float speedReward = RS_CLAMP(player.vel.Length() / 1500.f, 0.5f, 1.0f);

			float boostUsed = boostStart - player.boost;
			float boostPenalty = 0.f;
			if (boostUsed > BOOST_WASTE_LIMIT) {
				boostPenalty = (boostUsed - BOOST_WASTE_LIMIT) * BOOST_PENALTY_SCALE;
			}

			return (centeringReward * positionReward * speedReward) + boostPenalty;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !state.prev) return 0.0;

			int agentId = player.carId;
			if (agentState.find(agentId) == agentState.end()) {
				agentState[agentId] = 0;
				agentDribbleTicks[agentId] = 0;
				agentBoostAtDribbleStart[agentId] = player.boost;
			}

			int currentState = agentState[agentId];
			float currentBoostStart = agentBoostAtDribbleStart[agentId];

			// STATE 0: ON GROUND
			if (currentState == 0) {
				if (IsDribbling(player, state)) {
					agentState[agentId] = 1;
					agentDribbleTicks[agentId] = 1;
					agentBoostAtDribbleStart[agentId] = player.boost;
					return GetDribbleReward(player, state, player.boost) * DRIBBLE_REWARD_SCALE;
				}
			}
			// STATE 1: DRIBBLING
			else if (currentState == 1) {
				agentDribbleTicks[agentId]++;

				bool justJumped = player.prev->isOnGround && !player.isOnGround;
				if (justJumped) {
					agentState[agentId] = 2;
					return 0.0f;
				}
				else if (!IsDribbling(player, state)) {
					agentState[agentId] = 0;
					agentDribbleTicks[agentId] = 0;
					return 0.0f;
				}

				// Patience Logic
				float dribbleReward = GetDribbleReward(player, state, currentBoostStart);
				if (agentDribbleTicks[agentId] > MIN_DRIBBLE_TICKS_FOR_POP) {
					return dribbleReward * PATIENCE_BONUS_SCALE;
				}
				else {
					return dribbleReward * DRIBBLE_REWARD_SCALE;
				}
			}
			// STATE 2: AIRBORNE
			else if (currentState == 2) {
				if (player.isOnGround) {
					agentState[agentId] = 0;
					agentDribbleTicks[agentId] = 0;
					return GROUND_PENALTY;
				}
				if (player.hasFlipped) {
					agentState[agentId] = 0;
					agentDribbleTicks[agentId] = 0;
					return 0.0f;
				}
				if (state.ball.pos.z < 100) {
					agentState[agentId] = 0;
					agentDribbleTicks[agentId] = 0;
					return 0.0f;
				}

				bool justDoubleJumped = player.hasDoubleJumped && !player.prev->hasDoubleJumped;
				if (justDoubleJumped) {

					if (agentDribbleTicks[agentId] < MIN_DRIBBLE_TICKS_FOR_POP) {
						return -10.0f; // Punish early pop
					}

					float curVertVel = state.ball.vel.z;
					Vec ballHorizontalVel = Vec(state.ball.vel.x, state.ball.vel.y, 0);
					float horizontalSpeed = ballHorizontalVel.Length();

					float popAngle = atan2f(curVertVel, horizontalSpeed + 0.1f);
					const float idealAngle = 1.396f; // 80 degrees
					const float maxAngleError = 0.087f;

					float angleQuality = RS_MAX(0.f, 1.0f - (abs(popAngle - idealAngle) / maxAngleError));

					if (angleQuality < 0.2f) agentState[agentId] = 0;

					return POP_JACKPOT * angleQuality;
				}

				Vec dirToBall = (state.ball.pos - player.pos).Normalized();
				float tiltReward = player.rotMat.forward.Dot(dirToBall);
				return RS_MAX(0.f, tiltReward) * TILT_REWARD;
			}

			return 0.0f;
		}
	};



	/**
	 * @brief A generic flick reward structure.
	 * NOTE: Specific named flicks are too complex to detect without a dedicated state machine.
	 * This rewards any flick where the ball gains significant vertical velocity.
	 */
	class GenericFlickReward : public Reward {
	private:
		float lastBallVelZ = 0.0f;

	public:
		void Reset(const GameState& initialState) override {
			lastBallVelZ = initialState.ball.vel.z;
		}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float reward = 0.0f;

			// Detect flip/flick-like motion: off-ground and angular velocity spike
			bool flicking = !player.isOnGround && fabs(player.angVel.x) > 2.5f;

			if (player.ballTouchedStep && flicking) {
				float zVelChange = state.ball.vel.z - lastBallVelZ;

				// Reward sudden upward acceleration of the ball
				if (zVelChange > 300.f) {
					reward = RS_CLAMP(zVelChange / (CommonValues::CAR_MAX_SPEED / 2.f), 0.f, 3.f);
				}
			}

			// Cache current ball z velocity for next frame
			lastBallVelZ = state.ball.vel.z;
			return reward;
		}
	};

	class SwiftGroundDribbleReward : public Reward {
	private:
		struct DribbleState {
			bool isDribbling = false;
			float dribbleStartTime = 0.0f;
			float lastTouchTime = 0.0f;
			int consecutiveTouches = 0;
			Vec lastBallPos;
			Vec lastBallVel;
			float totalDribbleTime = 0.0f;
			float maxDribbleTime = 8.0f;
			float minDribbleTime = 1.0f;
			float maxDribbleDistance = 2000.0f;
			float minBallHeight = 50.0f;
		};

		std::unordered_map<uint32_t, DribbleState> dribbleStates;

		// Configuration parameters
		float maxDistance;
		float proximityWeight;
		float velocityWeight;
		float directionWeight;
		float ceilingBonus;
		float groundBonus;
		float touchBonus;
		float controlPenalty;
		float flickBonus;
		float ceilingDribbleBonus;
		float precisionBonus;
		float momentumBonus;

	public:
		SwiftGroundDribbleReward(
			float maxDistance = 300.0f,
			float proximityWeight = 1.0f,
			float velocityWeight = 0.5f,
			float directionWeight = 1.0f,
			float ceilingBonus = 1.0f,
			float groundBonus = 0.5f,
			float touchBonus = 0.2f,
			float controlPenalty = -0.1f,
			float flickBonus = 1.0f,
			float ceilingDribbleBonus = 1.5f,
			float precisionBonus = 0.5f,
			float momentumBonus = 0.3f
		) : maxDistance(maxDistance), proximityWeight(proximityWeight),
			velocityWeight(velocityWeight), directionWeight(directionWeight),
			ceilingBonus(ceilingBonus), groundBonus(groundBonus),
			touchBonus(touchBonus), controlPenalty(controlPenalty),
			flickBonus(flickBonus), ceilingDribbleBonus(ceilingDribbleBonus),
			precisionBonus(precisionBonus), momentumBonus(momentumBonus) {
		}

		virtual void Reset(const GameState& initialState) override {
			dribbleStates.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0.0f;

			// Optimized: use [] operator which auto-constructs, no need for find check
			uint32_t playerId = player.carId;
			DribbleState& dribbleState = dribbleStates[playerId];

			float reward = 0.0f;

			// Check if player touched the ball
			if (player.ballTouchedStep) {
				reward += touchBonus;
				dribbleState.consecutiveTouches++;
				dribbleState.lastTouchTime = 0.0f;

				// Detect flick
				if (dribbleState.isDribbling && dribbleState.consecutiveTouches > 1) {
					float flickVelocity = CalculateFlickVelocity(state, dribbleState);
					if (flickVelocity > 500.0f) {
						reward += flickBonus * (flickVelocity / 1000.0f);
					}
				}
			}
			else {
				dribbleState.consecutiveTouches = 0;
			}

			// Update dribble state
			UpdateDribbleState(player, state, dribbleState);

			// Calculate advanced skill score
			if (dribbleState.isDribbling) {
				float skillScore = CalculateAdvancedSkillScore(player, state, dribbleState);
				reward += skillScore;
			}

			// Check if dribble should end
			if (ShouldEndDribble(player, state, dribbleState)) {
				reward += CalculateFinalReward(dribbleState);
				ResetDribbleState(dribbleState);
			}

			return reward;
		}

	private:
		void UpdateDribbleState(const Player& player, const GameState& state, DribbleState& dribbleState) {
			// Optimized: compute toBall once, reuse for both Length and Normalized
			Vec toBall = state.ball.pos - player.pos;
			float distanceToBall = toBall.Length();
			Vec toBallNorm = (distanceToBall > 1e-6f) ? toBall / distanceToBall : Vec(0, 0, 0);

			// Check if we should start dribbling
			if (!dribbleState.isDribbling && distanceToBall < maxDistance && state.ball.pos.z < 200.0f) {
				dribbleState.isDribbling = true;
				dribbleState.dribbleStartTime = 0.0f;
				dribbleState.lastBallPos = state.ball.pos;
				dribbleState.lastBallVel = state.ball.vel;
			}

			// Update dribble time
			if (dribbleState.isDribbling) {
				dribbleState.dribbleStartTime += state.deltaTime;
				dribbleState.totalDribbleTime += state.deltaTime;
				dribbleState.lastTouchTime += state.deltaTime;
			}
		}

		float CalculateAdvancedSkillScore(const Player& player, const GameState& state, const DribbleState& dribbleState) {
			float score = 0.0f;

			// Optimized: reuse distanceToBall if available, or compute once
			Vec toBall = state.ball.pos - player.pos;
			float distanceToBall = toBall.Length();
			float proximityReward = proximityWeight * (1.0f - (distanceToBall / maxDistance));
			score += proximityReward;

			// Optimized: cache normalized vectors
			float ballVelLen = state.ball.vel.Length();
			Vec ballDirection = (ballVelLen > 1e-6f) ? state.ball.vel / ballVelLen : Vec(0, 0, 0);
			float playerVelLen = player.vel.Length();
			Vec playerDirection = (playerVelLen > 1e-6f) ? player.vel / playerVelLen : Vec(0, 0, 0);
			float velocityMatch = velocityWeight * playerDirection.Dot(ballDirection);
			score += velocityMatch;

			// Direction to goal
			Vec toGoalVec = Vec(0, 5200, 0) - state.ball.pos;
			float toGoalLen = toGoalVec.Length();
			Vec toGoal = (toGoalLen > 1e-6f) ? toGoalVec / toGoalLen : Vec(0, 0, 0);
			float directionBonus = directionWeight * ballDirection.Dot(toGoal);
			score += directionBonus;

			// Ceiling dribbling bonus
			if (state.ball.pos.z > 1000.0f) {
				score += ceilingDribbleBonus;
			}

			// Ground dribbling bonus
			if (state.ball.pos.z < 200.0f) {
				score += groundBonus;
			}

			// Precision bonus
			float precision = CalculateTouchPrecision(player, state);
			score += precisionBonus * precision;

			// Momentum conservation
			float momentum = CalculateMomentumConservation(player, state, dribbleState);
			score += momentumBonus * momentum;

			// Cap the total score to prevent excessive rewards
			return std::min(score, 2.0f);
		}

		float CalculateFlickVelocity(const GameState& state, const DribbleState& dribbleState) {
			Vec velocityChange = state.ball.vel - dribbleState.lastBallVel;
			return velocityChange.Length();
		}

		float CalculateTouchPrecision(const Player& player, const GameState& state) {
			// Optimized: cache normalized vectors
			float ballVelLen = state.ball.vel.Length();
			float playerVelLen = player.vel.Length();
			if (ballVelLen < 1e-6f || playerVelLen < 1e-6f) return 0.0f;

			Vec ballDirection = state.ball.vel / ballVelLen;
			Vec playerDirection = player.vel / playerVelLen;
			return playerDirection.Dot(ballDirection);
		}

		float CalculateMomentumConservation(const Player& player, const GameState& state, const DribbleState& dribbleState) {
			// Calculate how well the player maintains momentum during dribbling
			if (dribbleState.consecutiveTouches < 2) return 0.0f;

			// Optimized: reuse already computed lengths if available
			float currentSpeed = player.vel.Length();
			float ballSpeed = state.ball.vel.Length();
			return std::min(currentSpeed / 1000.0f, ballSpeed / 1000.0f);
		}

		bool ShouldEndDribble(const Player& player, const GameState& state, const DribbleState& dribbleState) {
			if (!dribbleState.isDribbling) return false;

			// Optimized: use LengthSq for distance check
			Vec toBall = state.ball.pos - player.pos;
			float distanceToBallSq = toBall.LengthSq();
			float maxDistSq = maxDistance * maxDistance;
			if (distanceToBallSq > maxDistSq) return true;

			// End if ball is too high (lost control)
			if (state.ball.pos.z > 500.0f) return true;

			// End if dribble time exceeded
			if (dribbleState.totalDribbleTime > dribbleState.maxDribbleTime) return true;

			// End if no touches for too long
			if (dribbleState.lastTouchTime > 2.0f) return true;

			return false;
		}

		float CalculateFinalReward(const DribbleState& dribbleState) {
			if (dribbleState.totalDribbleTime < dribbleState.minDribbleTime) {
				return controlPenalty;
			}

			float timeBonus = dribbleState.totalDribbleTime / dribbleState.maxDribbleTime;
			float touchBonus = dribbleState.consecutiveTouches * 0.1f; // Reduced from 0.5f

			return timeBonus + touchBonus;
		}

		void ResetDribbleState(DribbleState& dribbleState) {
			dribbleState.isDribbling = false;
			dribbleState.dribbleStartTime = 0.0f;
			dribbleState.consecutiveTouches = 0;
			dribbleState.totalDribbleTime = 0.0f;
			dribbleState.lastTouchTime = 0.0f;
		}
	};

	// Named flicks just inherit from the generic one for this implementation
	class MawkyzFlickReward : public GenericFlickReward {};
	class EnhancedDiagonalFlickReward : public GenericFlickReward {};

	/**
	 * @brief Rewards making a defensive touch on the backboard.
	 */
	class BackboardDefenseReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.ballTouchedStep) return 0;

			bool is_on_own_backboard =
				(abs(player.pos.y) > CommonValues::BACK_WALL_Y - 400) &&
				(player.pos.z > CommonValues::GOAL_HEIGHT) &&
				(RS_SGN(player.pos.y) == (player.team == Team::ORANGE ? 1 : -1));

			if (is_on_own_backboard) {
				Vec own_goal_pos = player.team == Team::BLUE ? CommonValues::BLUE_GOAL_BACK : CommonValues::ORANGE_GOAL_BACK;
				Vec dir_from_goal = (state.ball.pos - own_goal_pos).Normalized();

				// Reward clearing the ball away from goal
				return state.ball.vel.Dot(dir_from_goal) / CommonValues::BALL_MAX_SPEED;
			}

			return 0;
		}
	};

	class kuesresetreward : public Reward {
	public:
		float minHeight;
		float maxDist;

		kuesresetreward(float minHeight = 150.f, float maxDist = 300.f) :
			minHeight(minHeight), maxDist(maxDist) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround || player.HasFlipOrJump()) {
				return 0.f;
			}

			if (player.pos.z < minHeight) {
				return 0.f;
			}

			if (player.rotMat.up.z >= 0) {
				return 0.f;
			}

			float distToBall = (player.pos - state.ball.pos).Length();
			if (distToBall > maxDist) {
				return 0.f;
			}

			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec relVel = player.vel - state.ball.vel;
			float approachSpeed = relVel.Dot(dirToBall);
			if (approachSpeed <= 0) {
				return 0.f;
			}

			float normSpeed = RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
			float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
			float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

			return std::min({ normSpeed, normAlign, normDist });
		}
	};


	class BouncyAirDribbleReward : public Reward {
	public:
		float bouncyDistThreshold;
		float bouncyVelThreshold;
		float minHeightForDribble;
		float minSpeedTowardGoal;
		float maxDistFromBall;
		float minPlayerToBallUpwards;
		float nearGoalShutoffTime;

		BouncyAirDribbleReward(
			float bouncyDist = 10.f, float bouncyVel = 25.f, float minHeight = 200.f,
			float minSpeed = 500.f, float maxDist = 300.f, float minPlayerToBallZ = 0.1f,
			float shutoffTime = 1.f) :
			bouncyDistThreshold(bouncyDist), bouncyVelThreshold(bouncyVel), minHeightForDribble(minHeight),
			minSpeedTowardGoal(minSpeed), maxDistFromBall(maxDist), minPlayerToBallUpwards(minPlayerToBallZ),
			nearGoalShutoffTime(shutoffTime) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {

			Vec opponentGoal = (player.team == Team::BLUE) ?
				CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;

			bool isBouncy = false;
			float distToBall = (player.pos - state.ball.pos).Length();
			if (distToBall > bouncyDistThreshold) {
				isBouncy = true;
			}
			else {
				float relVelMag = (player.vel - state.ball.vel).Length();
				if (relVelMag > bouncyVelThreshold) {
					isBouncy = true;
				}
			}

			if (!isBouncy) {
				return 0.f;
			}

			bool heightConditionMet = false;
			if (state.ball.pos.z > minHeightForDribble) {
				heightConditionMet = true;
			}
			else if (player.vel.z > 0 && state.ball.vel.z > 0) {
				heightConditionMet = true;
			}

			if (!heightConditionMet) return 0.f;

			if (distToBall > maxDistFromBall) return 0.f;

			float distToGoal2D = (player.pos - opponentGoal).Length2D();
			float minBoostRequired = 15.f * std::pow(distToGoal2D / 5000.f, 1.75f);
			if (player.boost < minBoostRequired) return 0.f;

			Vec playerToBallDir = (state.ball.pos - player.pos).Normalized();
			if (playerToBallDir.z < minPlayerToBallUpwards) return 0.f;

			Vec ballToGoalDir = (opponentGoal - state.ball.pos).Normalized();
			if (player.vel.Dot(ballToGoalDir) < minSpeedTowardGoal || state.ball.vel.Dot(ballToGoalDir) < minSpeedTowardGoal) {
				return 0.f;
			}

			Vec playerToBallDir2D = Vec(playerToBallDir.x, playerToBallDir.y, 0).Normalized();
			Vec ballToGoalDir2D = Vec(ballToGoalDir.x, ballToGoalDir.y, 0).Normalized();
			if (playerToBallDir2D.Dot(ballToGoalDir2D) < 0.75) return 0.f; //near the goal value

			float yDistToGoal = std::abs(state.ball.pos.y - opponentGoal.y);
			float timeToGoal = yDistToGoal / std::max(1.f, std::abs(state.ball.vel.y));

			if (timeToGoal > 0) {
				float gravityZ = state.lastArena ? state.lastArena->GetMutatorConfig().gravity.z : -650.f;
				float heightAtGoal = state.ball.pos.z + state.ball.vel.z * timeToGoal + 0.5f * gravityZ * timeToGoal * timeToGoal;
				if (heightAtGoal > CommonValues::GOAL_HEIGHT + 100) return 0.f;
			}

			if (timeToGoal < nearGoalShutoffTime) {
				return 0.f;
			}

			return 1.f;
		}
	};

	class ApproachForDemoHelperReward2 : public Reward {
	private:
		float angleWeight;
		float distanceWeight;

	public:
		// Constructor Implementation
		ApproachForDemoHelperReward2(float angle_weight = 0.5f, float distance_weight = 0.5f)
			: angleWeight(angle_weight), distanceWeight(distance_weight) {
		}

		// Reset Implementation
		virtual void Reset(const GameState& initialState) override {}

		// GetReward Implementation with LESS SPAMMY Logging (Only prints significant rewards)
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float reward = 0.0f;

			Vec myPos = player.pos;
			Vec myVel = player.vel;
			Vec myFwd = player.rotMat.forward;
			Team myTeam = player.team;

			const Player* closestOpponent = nullptr;
			float closestDist = FLT_MAX;

			// Find nearest opponent
			for (const auto& opponent : state.players) {
				if (opponent.team == myTeam || opponent.carId == player.carId) continue;
				float dist = (opponent.pos - myPos).Length();
				if (dist < closestDist) {
					closestDist = dist;
					closestOpponent = &opponent;
				}
			}

			if (closestOpponent) {
				Vec toEnemy = closestOpponent->pos - myPos;
				float distance = toEnemy.Length();

				if (distance < 0.001f) return 0.0f;

				Vec toEnemyDir = toEnemy / distance;
				float myFwdLength = myFwd.Length();
				if (myFwdLength < 0.001f) return 0.0f;
				Vec myFwdDir = myFwd / myFwdLength;

				float mySpeed = myVel.Length();

				float alignment = myFwdDir.Dot(toEnemyDir);
				float angleReward = std::max(0.0f, alignment);

				float distanceReward = std::max(0.0f, 1.0f - (distance / 5000.0f));

				float speed_scale = mySpeed / 2300.0f;
				reward = (angleWeight * angleReward + distanceWeight * distanceReward) * speed_scale;

				// *** LOGGING CONDITION CHANGE: Only print if reward is above a significant threshold (e.g., 0.5) ***
				// This filters out tiny rewards the agent gets every tick for vaguely facing the right way.
				if (reward >= 1.2f) {
					std::cout << "[ApproachReward] 🎯 HIGH APPROACH: P" << player.carId
						<< " towards O" << closestOpponent->carId
						<< " | Dist: " << distance
						<< " | Speed: " << mySpeed
						<< " | Reward: " << reward << std::endl;
				}
			}

			return reward;
		}
	};

	// Optional cosine similarity helper
	inline float cosine_similarity(const Vec& a, const Vec& b)
	{
		float dotProduct = a.Dot(b);
		float lenA = a.Length();
		float lenB = b.Length();
		if (lenA == 0 || lenB == 0)
			return 0.0f;
		return dotProduct / (lenA * lenB);
	}

	inline Vec Normalize(const Vec& v) {
		float len = v.Length();
		if (len == 0) return v;
		return v / len;
	}

	inline float Dot(const Vec& a, const Vec& b) {
		return a.Dot(b);
	}



	/**
	 * @brief A complex reward for performing air dribbles.
	 * It rewards being close to the ball in the air, maintaining a controlled speed,
	 * touching the ball multiple times, and using air roll.
	 */
	class AirDribbleWithRollReward : public Reward {
	public:
		AirDribbleWithRollReward(
			float maxHeight = 350.0f,
			float maxDistance = 250.0f,
			float proximityWeight = 3.0f,
			float velocityWeight = 3.0f,
			float heightWeight = 1.0f,
			float touchBonus = 6.0f,
			float boostThreshold = 0.2f,
			float heightTarget = 1000.0f,
			float heightTargetWeight = 1.0f,
			float upwardBonus = 0.4f,
			float airRollWeight = 299.75f,
			float speedWeight = 1.0f,
			float boostEfficiencyWeight = 1.2f,
			float goalDirectionWeight = 20.5f, // Bonus for moving towards opponent goal
			float awayFromGoalPenaltyWeight = 5.f // Penalty for moving away from opponent goal
		) : minHeight(minHeight), maxDistance(maxDistance),
			proximityWeight(proximityWeight), velocityWeight(velocityWeight),
			heightWeight(heightWeight), touchBonus(touchBonus),
			boostThreshold(boostThreshold), heightTarget(heightTarget),
			heightTargetWeight(heightTargetWeight), upwardBonus(upwardBonus),
			airRollWeight(airRollWeight), speedWeight(speedWeight),
			boostEfficiencyWeight(boostEfficiencyWeight),
			goalDirectionWeight(goalDirectionWeight),
			awayFromGoalPenaltyWeight(awayFromGoalPenaltyWeight) {
		}

		inline virtual void Reset(const GameState& initialState) override {
			playerStates.clear();
			for (const auto& player : initialState.players) {
				playerStates[player.carId] = { player.ballTouchedTick, 0, 0.0f, 0.0f };
			}
		}

		inline virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (playerStates.find(player.carId) == playerStates.end()) {
				playerStates[player.carId] = { player.ballTouchedTick, 0, 0.0f, 0.0f };
			}
			auto& pState = playerStates[player.carId];

			auto resetState = [&]() {
				pState.airTouchStreak = 0;
				pState.lastBallTouch = player.ballTouchedStep;
				pState.lastRollInput = 0.0f;
				pState.directionChanges = 0.0f;
				return 0.0f;
				};

			bool isAirborne = !player.isOnGround &&
				player.pos.z >= minHeight &&
				state.ball.pos.z >= minHeight;

			if (!isAirborne) return resetState();

			Vec distVec = state.ball.pos - player.pos;
			float dist = distVec.Length();
			if (dist > maxDistance) return resetState();

			float proximityFactor = std::max(0.0f, 1.0f - (dist - CommonValues::BALL_RADIUS) / std::max(1e-6f, maxDistance - CommonValues::BALL_RADIUS));
			float proxReward = proximityFactor * proximityWeight;

			float velReward = 0.0f;
			if (dist > 1e-6f) {
				Vec dirToBall = distVec / dist;
				float speedTowardsBall = player.vel.Dot(dirToBall);
				if (speedTowardsBall > 0) {
					velReward = (speedTowardsBall / CommonValues::CAR_MAX_SPEED) * velocityWeight;
				}
			}

			float heightFactor = std::max(0.0f, (player.pos.z - minHeight) / std::max(1e-6f, CommonValues::CEILING_Z - minHeight));
			float heightReward = heightFactor * heightWeight;

			float heightDiff = abs(player.pos.z - heightTarget);
			float heightScale = std::max(0.0f, 1.0f - heightDiff / heightTarget);
			float heightTargetReward = heightScale * heightTargetWeight;

			float controlledSpeedReward = 0.0f;
			float playerSpeed = player.vel.Length();
			constexpr float OPTIMAL_MIN_SPEED = 800.0f;
			constexpr float OPTIMAL_MAX_SPEED = 1400.0f;
			if (playerSpeed >= OPTIMAL_MIN_SPEED && playerSpeed <= OPTIMAL_MAX_SPEED) {
				controlledSpeedReward = speedWeight;
			}
			else if (playerSpeed > OPTIMAL_MAX_SPEED) {
				float excessSpeedPenalty = std::min(0.5f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
				controlledSpeedReward = -excessSpeedPenalty * speedWeight;
			}

			float boostConservationReward = 0.0f;
			if (player.boost > 30.f) {
				float boostFactor = powf(player.boost / 100.f, 0.7f);
				boostConservationReward = boostFactor * boostEfficiencyWeight;
			}

			Vec targetPos = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec dirToGoal = (targetPos - player.pos).Normalized();
			float goalAlignment = dirToGoal.Dot(player.vel.Normalized());

			// **Bonus for dribbling towards the opponent's goal**
			float goalDirectionReward = std::max(0.f, goalAlignment) * goalDirectionWeight;

			// **Penalty for dribbling away from the opponent's goal**
			float awayPenalty = std::min(0.f, goalAlignment) * awayFromGoalPenaltyWeight; // awayFromGoalPenaltyWeight should be positive

			float baseReward = proxReward + velReward + heightReward + heightTargetReward + controlledSpeedReward + boostConservationReward + goalDirectionReward + awayPenalty;
			float totalWeight = proximityWeight + velocityWeight + heightWeight + heightTargetWeight + speedWeight + boostEfficiencyWeight + goalDirectionWeight + awayFromGoalPenaltyWeight;

			float airRollReward = 0.0f;
			float currentRollInput = player.prevAction.roll;

			if (abs(currentRollInput) > 0.1f && abs(pState.lastRollInput) > 0.1f) {
				if (currentRollInput * pState.lastRollInput < 0) {
					pState.directionChanges += 1.0f;
				}
			}

			if (pState.directionChanges > 0) {
				pState.directionChanges = std::max(0.0f, pState.directionChanges - 0.05f);
			}

			pState.lastRollInput = currentRollInput;
			bool isWiggling = pState.directionChanges > 2.0f;

			if (!isWiggling && abs(currentRollInput) > 0.1f) {
				float baseRollReward = airRollWeight;
				if (currentRollInput > 0.1f) baseRollReward *= 1.5f;
				else baseRollReward *= 0.3f;

				if (player.prevAction.boost > 0 && player.boost > (boostThreshold * 100)) {
					baseRollReward *= 1.4f;
				}
				airRollReward = baseRollReward;
			}

			baseReward += airRollReward;
			totalWeight += 1.0f;

			float normalizedBaseReward = baseReward / std::max(1e-6f, totalWeight);
			float finalReward = normalizedBaseReward;

			if (player.vel.z > 0) finalReward += upwardBonus;

			Vec carUp = player.rotMat.up;
			bool belowBall = player.pos.z < state.ball.pos.z - 30;
			bool facingBall = dist > 1e-6f && distVec.Dot(carUp) / dist > 0.7f;
			if (belowBall && facingBall) finalReward += 0.2f;

			if (!pState.lastBallTouch && player.ballTouchedStep) {
				pState.airTouchStreak++;
				float touchMultiplier = 1.0f;

				if (playerSpeed >= OPTIMAL_MIN_SPEED && playerSpeed <= OPTIMAL_MAX_SPEED) {
					touchMultiplier *= 2.5f;
				}
				else if (playerSpeed > OPTIMAL_MAX_SPEED) {
					float speedPenalty = std::min(0.7f, (playerSpeed - OPTIMAL_MAX_SPEED) / 1000.0f);
					touchMultiplier *= std::max(0.1f, 1.5f - speedPenalty);
				}
				else {
					touchMultiplier *= 1.2f;
				}

				if (player.boost > (boostThreshold * 100)) touchMultiplier *= 1.8f;
				if (dist < CommonValues::BALL_RADIUS + 150.0f) touchMultiplier *= 1.4f;

				float touchReward = touchBonus * pState.airTouchStreak * touchMultiplier;
				finalReward += touchReward;
			}

			pState.lastBallTouch = player.ballTouchedStep;

			float heightScaling = state.ball.pos.z / 1000.0f;
			finalReward *= std::max(0.1f, std::min(2.0f, heightScaling));

			return finalReward;
		}

	private:
		struct PlayerState {
			bool lastBallTouch;
			int airTouchStreak;
			float lastRollInput;
			float directionChanges;
		};

		std::map<uint32_t, PlayerState> playerStates;

		// Reward parameters
		float minHeight;
		float maxDistance;
		float proximityWeight;
		float velocityWeight;
		float heightWeight;
		float touchBonus;
		float boostThreshold;
		float heightTarget;
		float heightTargetWeight;
		float upwardBonus;
		float airRollWeight;
		float speedWeight;
		float boostEfficiencyWeight;
		float goalDirectionWeight;
		float awayFromGoalPenaltyWeight;
	};

	class AirTouchReward : public Reward {
	public:
		const float MAX_TIME_IN_AIR = 1.75f;

		AirTouchReward() {}

		virtual void Reset(const GameState& initialState) override {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.ballTouchedStep) {
				float airTimeFraction = std::min(player.airTime, MAX_TIME_IN_AIR) / MAX_TIME_IN_AIR;
				float heightFraction = state.ball.pos.z / CommonValues::CEILING_Z;

				return std::min(airTimeFraction, heightFraction);
			}
			return 0.0f;
		}
	};


	class ContinuousFlipResetReward : public Reward {
	public:
		float minHeight;
		float maxDist;

		ContinuousFlipResetReward(float minHeight = 150.f, float maxDist = 300.f) :
			minHeight(minHeight), maxDist(maxDist) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.isOnGround || player.HasFlipOrJump()) {
				return 0.f;
			}

			if (player.pos.z < minHeight) {
				return 0.f;
			}

			if (player.rotMat.up.z >= 0) {
				return 0.f;
			}

			float distToBall = (player.pos - state.ball.pos).Length();
			if (distToBall > maxDist) {
				return 0.f;
			}

			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec relVel = player.vel - state.ball.vel;
			float approachSpeed = relVel.Dot(dirToBall);
			if (approachSpeed <= 0) {
				return 0.f;
			}

			float normSpeed = RS_CLAMP(approachSpeed / CommonValues::CAR_MAX_SPEED, 0.f, 1.f);
			float normAlign = ((-player.rotMat.up).Dot(dirToBall) + 1.f) / 2.f;
			float normDist = 1.f - RS_CLAMP(distToBall / maxDist, 0.f, 1.f);

			// Calculate the reward
			float reward = std::min({ normSpeed, normAlign, normDist });
			return reward;
		}
	};

	class TouchBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.ballTouchedStep;
		}
	};

	class FlipResetEventReward : public Reward {
	public:
		float minHeight;
		float maxDist;
		float minUpZ;

		FlipResetEventReward(float minHeight = 150.f, float maxDist = 150.f, float minUpZ = -0.7f) :
			minHeight(minHeight), maxDist(maxDist), minUpZ(minUpZ) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev) return 0.f;

			bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() && !player.prev->HasFlipOrJump();

			if (gotReset) {

				if (player.pos.z > minHeight && player.rotMat.up.z < minUpZ && (player.pos - state.ball.pos).Length() < maxDist) {
					float reward = 1.f;
					// Print to console because the reward is positive
					std::cout << "FlipResetEventReward: " << reward << std::endl;
					return reward;
				}
			}

			return 0.f;
		}
	};

	class DribbleAirdribbleBumpDemoReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			float reward = 0.f;
			if (state.ball.pos.y > 4600 && std::abs(state.ball.pos.x) < GOAL_WIDTH_FROM_CENTER && state.ball.vel.y > 0) {
				if (player.pos.y > state.ball.pos.y && std::abs(player.pos.x) < GOAL_WIDTH_FROM_CENTER) {
					reward += 0.001f;
					if (player.eventState.bump) {
						reward += 0.5f * (state.ball.vel.y / 1000);
					}
					if (player.eventState.demo) {
						reward += 1.0f * (state.ball.vel.y / 1000);
					}
				}
			}
			return reward;
		}
	};

	class ResetShotReward : public Reward {
	private:

		std::map<uint32_t, uint64_t> _tickCountWhenResetObtained;

	public:
		ResetShotReward() {}

		virtual void Reset(const GameState& initial_state) override {
			_tickCountWhenResetObtained.clear();
		}

		virtual void PreStep(const GameState& state) override {
			if (!state.lastArena) return;
			for (const auto& player : state.players) {
				if (!player.prev) continue;

				bool gotReset = !player.prev->isOnGround && player.HasFlipOrJump() && !player.prev->HasFlipOrJump();
				if (gotReset) {
					_tickCountWhenResetObtained[player.carId] = state.lastArena->tickCount;
				}
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev || !state.prev) return 0.f;

			auto it = _tickCountWhenResetObtained.find(player.carId);
			if (it == _tickCountWhenResetObtained.end()) {
				return 0.f;
			}

			bool flipWasUsedForTouch = player.ballTouchedStep &&
				!player.isOnGround &&
				!player.hasJumped &&
				player.prev->HasFlipOrJump() &&
				!player.HasFlipOrJump();

			if (flipWasUsedForTouch) {

				float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
				float ballSpeed = state.ball.vel.Length();
				float baseReward = (hitForce + ballSpeed) / (CommonValues::CAR_MAX_SPEED + CommonValues::BALL_MAX_SPEED);

				uint64_t ticksSinceReset = state.lastArena->tickCount - it->second;
				float timeSinceReset = ticksSinceReset * CommonValues::TICK_TIME;
				float timeBonus = 1.f + std::log1p(timeSinceReset);

				_tickCountWhenResetObtained.erase(it);

				// Calculate the reward
				float reward = baseReward * timeBonus;

				// Print to console if the reward is positive
				if (reward > 0) {
					std::cout << "ResetShotReward: " << reward << std::endl;
				}

				return reward;
			}

			if (player.isOnGround) {
				_tickCountWhenResetObtained.erase(it);
			}

			return 0.f;
		}
	};

	/**
	 * @brief A stateful reward for air dribbling while using air roll.
	 * It gives rewards for maintaining proximity to the ball, moving towards it,
	 * gaining height, staying at a target height, conserving boost, and using air roll effectively.
	 * It also gives a large bonus for each consecutive touch in the air.
	 */
	class AirDribbleWithRollRewardImpl : public Reward {
	public:
		// Configurable parameters
		float min_height = 326.0f;
		float max_distance = 250.0f;
		float proximity_weight = 3.0f;
		float velocity_weight = 3.0f;
		float height_weight = 1.0f;
		float touch_bonus = 3.0f;
		float boost_threshold = 20.0f; // Boost amount / 100
		float height_target = 1000.0f;
		float height_target_weight = 1.0f;
		float upward_bonus = 0.1f;
		float air_roll_weight = 0.5f;
		float speed_weight = 1.0f;
		float boost_efficiency_weight = 1.2f;

	private:
		// Per-player state
		bool last_ball_touch = false;
		int air_touch_streak = 0;
		float last_roll_input = 0.0f;
		int direction_changes = 0;

	public:
		virtual void Reset(const GameState& initialState) override {
			last_ball_touch = false;
			air_touch_streak = 0;
			last_roll_input = 0.0f;
			direction_changes = 0;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Reset streak and state if on ground or conditions are not met
			bool is_airborne = !player.isOnGround && player.pos.z >= min_height && state.ball.pos.z >= min_height;
			float dist = player.pos.Dist(state.ball.pos);
			bool is_close = dist <= max_distance;

			if (!is_airborne || !is_close) {
				air_touch_streak = 0;
				last_ball_touch = player.ballTouchedStep;
				last_roll_input = 0.0f;
				direction_changes = 0;
				return 0.0f;
			}

			// --- Base Reward Components ---
			float prox_reward = (1.0f - (dist - CommonValues::BALL_RADIUS) / (max_distance - CommonValues::BALL_RADIUS)) * proximity_weight;

			Vec dir_to_ball = (state.ball.pos - player.pos).Normalized();
			float speed_towards_ball = player.vel.Dot(dir_to_ball);
			float vel_reward = (speed_towards_ball > 0) ? (speed_towards_ball / CommonValues::CAR_MAX_SPEED) * velocity_weight : 0;

			float height_reward = ((player.pos.z - min_height) / (CommonValues::CEILING_Z - min_height)) * height_weight;

			float height_diff = abs(player.pos.z - height_target);
			float height_target_reward = (1.0f - height_diff / height_target) * height_target_weight;

			float controlled_speed_reward = 0.0f;
			float player_speed = player.vel.Length();
			if (player_speed >= 800.0f && player_speed <= 1400.0f) {
				controlled_speed_reward = speed_weight;
			}
			else if (player_speed > 1400.0f) {
				controlled_speed_reward = -std::min(0.5f, (player_speed - 1400.0f) / 1000.0f) * speed_weight;
			}

			float boost_conservation_reward = (player.boost > 30) ? (powf(player.boost / 100.f, 0.7f) * boost_efficiency_weight) : 0;

			// --- Air Roll Reward ---
			float air_roll_reward = 0.0f;
			float current_roll_input = player.prevAction.roll;

			if (abs(current_roll_input) > 0.1f && abs(last_roll_input) > 0.1f) {
				if (current_roll_input * last_roll_input < 0) { // Sign change
					direction_changes += 1;
				}
			}
			if (direction_changes > 0) direction_changes = std::max(0, direction_changes - 1); // Decay
			last_roll_input = current_roll_input;

			bool is_wiggling = direction_changes > 4; // Check for excessive wiggling
			if (!is_wiggling && abs(current_roll_input) > 0.1f) {
				air_roll_reward = air_roll_weight;
			}

			// --- Combine Rewards ---
			float base_reward = prox_reward + vel_reward + height_reward + height_target_reward + controlled_speed_reward + boost_conservation_reward + air_roll_reward;
			float total_weight = proximity_weight + velocity_weight + height_weight + height_target_weight + speed_weight + boost_efficiency_weight + 1.0f;
			float final_reward = base_reward / total_weight;

			if (player.vel.z > 0) final_reward += upward_bonus;

			// --- Touch Bonus ---
			if (!last_ball_touch && player.ballTouchedStep) {
				air_touch_streak += 1;
				float touch_multiplier = 1.0f;
				if (player_speed >= 800.0f && player_speed <= 1400.0f) touch_multiplier *= 2.5f;
				if (player.boost > boost_threshold) touch_multiplier *= 1.8f;
				if (dist < CommonValues::BALL_RADIUS + 150.0f) touch_multiplier *= 1.4f;

				final_reward += touch_bonus * air_touch_streak * touch_multiplier;
			}

			last_ball_touch = player.ballTouchedStep;

			// Final scaling based on height
			final_reward *= std::max(0.1f, std::min(2.0f, state.ball.pos.z / 1000.0f));

			return final_reward;
		}
	};

	/**
	 * @brief Rewards the ball for having high velocity towards the opponent's goal.
	 */
	class SpeedBallToGoalReward : public Reward {
	public:
		bool ownGoal = false;
		SpeedBallToGoalReward(bool ownGoal = false) : ownGoal(ownGoal) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal)
				targetOrangeGoal = !targetOrangeGoal;

			Vec targetPos = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;

			Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
			return state.ball.vel.Dot(dirToGoal) / CommonValues::BALL_MAX_SPEED;
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class VelocityReward : public Reward {
	public:
		bool isNegative;
		VelocityReward(bool isNegative = false) : isNegative(isNegative) {}
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED * (1 - 2 * isNegative);
		}
	};

	class EnergyReward : public Reward {
	public:
		const double GRAVITY = 650;
		const double MASS = 180;
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			auto max_energy = (MASS * GRAVITY * (CEILING_Z - 17.)) + (0.5 * MASS * (CAR_MAX_SPEED * CAR_MAX_SPEED));
			double energy = 0;
			double velocity = player.vel.Length();

			if (player.HasFlipOrJump()) {
				energy += 0.35 * MASS * (292 * 292);
			}
			if (player.HasFlipOrJump() and !player.isOnGround) {
				double dodge_impulse = (velocity <= 1700) ? (500 + (velocity / 17)) : (600 - (velocity - 1700));
				dodge_impulse = std::max(dodge_impulse - 25, 0.0);
				energy += 0.9 * 0.5 * MASS * (dodge_impulse * dodge_impulse);
				energy += 0.35 * MASS * 550. * 550.;
			}
			//height
			energy += MASS * GRAVITY * (player.pos.z - 17.) * 0.75; // fudge factor to reduce height
			//KE
			energy += 0.5 * MASS * velocity * velocity;
			//boost
			energy += 7.97e5 * player.boost;
			double norm_energy = player.isDemoed ? 0.0f : (energy / max_energy);
			return norm_energy;
		}
	};
	
	
	
	
	class FlyToGoalKeepHigh : public Reward {
public:
	// ============================================================================
	// PARAMETRI BASE
	// ============================================================================

	// --- Gating & reward base (per-tick) ---
	float min_air_height = 320.0f;
	float max_ball_dist = 650.0f;
	float per_tick_scale = 0.08f;

	// --- Touch bonus (esponenziale verso porta) ---
	float touch_base = 10.0f;
	float lambda_goal = 5.0f;
	float min_touch_factor = 0.0f;
	float max_delta_v = 2500.0f;
	bool  use_impulse_delta = true;

	// --- Air roll bonus (BILANCIATO) ---
	float airroll_bonus_base = 0.2f;
	float airroll_bonus_max = 1.6f;
	float min_airroll_rate = 0.02f;
	float airroll_speed_threshold = 150.0f;

	int   max_progression_ticks = 40;
	int   min_ticks_for_bonus = 10;

	int   decay_rate_active = 4;
	int   decay_rate_slow = 3;
	int   decay_rate_gating = 12;

	float smooth_factor_active = 0.70f;
	float smooth_factor_inactive = 0.40f;

	float quality_threshold = 0.8f;
	float quality_multiplier = 2.2f;
	float consistency_bonus_rate = 0.7f;
	int   consistency_start_ticks = 15;

	float direction_detect_mult = 2.0f;

	// --- Field position scaling ---
	float min_field_mult = 0.35f;
	float max_field_mult = 1.35f;
	float optimal_start_dist = 6000.0f;
	float too_far_penalty_dist = 10000.0f;
	bool  use_field_scaling = true;

	// --- Bonus "sotto la palla" (solo per touch) ---
	float under_ball_min_gap = 180.0f;
	float under_ball_max_gap = 450.0f;
	float under_touch_boost = 0.5f;

	// ============================================================================
	// PARAMETRI ALLINEAMENTO DIREZIONE TARGET
	// ============================================================================
	float w_dir_ball = 0.35f;
	float w_dir_goal = 0.75f;
	float min_goal_push_dot = 0.30f;
	float cross_track_penalty = 0.20f;
	float opposite_dir_penalty = 0.35f;

	// ============================================================================
	// PARAMETRI SHOT QUALITY
	// ============================================================================
	float shot_quality_bonus = 8.0f;
	float optimal_approach_angle = 25.0f;
	float shot_quality_distance_max = 0.8f;
	float optimal_shot_height = 380.0f;
	float shot_height_bonus = 4.0f;

	// ============================================================================
	// 🆕 OPTIMAL HEIGHT MAINTENANCE SYSTEM
	// ============================================================================
	float optimalHeight = 600.0f;              // Target height to maintain (uu)
	float optimalHeightTolerance = 50.0f;     // Tolerance range (+/- from optimal)
	float heightMaintenanceWeight = 0.2f;      // Weight for height maintenance reward
	float zVelMaintenanceWeight = 0.1f;        // Weight for Z velocity maintenance
	float targetZVel = 500.0f;                 // Target Z velocity (positive = rising, 0 = floating)
	float zVelTolerance = 300.0f;              // Tolerance for Z velocity
	float proximityWeight = 0.05f;              // Weight for proximity to ball
	float maxProximityDistance = 200.0f;       // Maximum distance for proximity bonus
	int   maxConsecutiveTouches = 20;          // Maximum consecutive touches tracked
	
	// Height relative to ball
	bool  useRelativeHeight = true;            // If true, optimal height is relative to ball
	float relativeHeightOffset = -80.0f;       // Offset from ball height (negative = below ball)
	float minAbsoluteHeight = 400.0f;          // Minimum absolute height even in relative mode
	
	// Height reward shaping
	float heightPenaltyScale = 0.3f;           // Penalty multiplier when outside optimal range
	float maxHeightDeviation = 500.0f;         // Max deviation before zero height reward
	float risingBonus = 0.1f;                  // Bonus for rising toward optimal height
	float maintainBonus = 0.2f;                // Bonus for maintaining optimal height

	// ============================================================================
	// STATE TRACKING
	// ============================================================================
	struct PlayerAirRollState {
		int   consecutive_ticks = 0;
		float prev_roll_direction = 0.0f;
		float smoothed_roll_rate = 0.0f;
		float accumulated_bonus = 0.0f;
		int   stable_direction_ticks = 0;
		float total_airroll_time = 0.0f;
		bool  has_reached_min_ticks = false;

		// Shot quality tracking
		float best_shot_quality = 0.0f;
		int   ticks_in_good_position = 0;

		// 🆕 Height maintenance tracking
		int   ticks_at_optimal_height = 0;
		float smoothed_z_vel = 0.0f;
		float prev_height = 0.0f;
		int   consecutive_touches = 0;
	};
	std::unordered_map<uint32_t, PlayerAirRollState> player_states;

	// ============================================================================
	// HELPER FUNCTIONS
	// ============================================================================

	inline static float clamp01(float x) {
		return std::max(0.0f, std::min(1.0f, x));
	}

	std::string GetName() override {
		return "FlyToGoalKeepHigh";
	}

	void Reset(const GameState& initialState) override {
		player_states.clear();
	}

	// 🆕 Calculate optimal target height (absolute or relative to ball)
	float GetTargetHeight(float ball_height) {
		if (useRelativeHeight) {
			float relative_target = ball_height + relativeHeightOffset;
			return std::max(minAbsoluteHeight, relative_target);
		}
		return optimalHeight;
	}

	// 🆕 Calculate height maintenance reward
	float GetHeightMaintenanceReward(const Player& player, float ball_height, PlayerAirRollState& ar_state) {
		float reward = 0.0f;
		float car_height = player.pos.z;
		float target_height = GetTargetHeight(ball_height);
		
		// Height deviation from optimal
		float height_diff = std::abs(car_height - target_height);
		
		// Smooth Z velocity tracking
		float current_z_vel = player.vel.z;
		ar_state.smoothed_z_vel = 0.8f * ar_state.smoothed_z_vel + 0.2f * current_z_vel;
		
		// ========================================
		// HEIGHT POSITION REWARD
		// ========================================
		if (height_diff <= optimalHeightTolerance) {
			// Within optimal range - full reward
			float height_quality = 1.0f - (height_diff / optimalHeightTolerance);
			reward += heightMaintenanceWeight * height_quality;
			ar_state.ticks_at_optimal_height++;
			
			// Milestone bonuses for maintaining height
			if (ar_state.ticks_at_optimal_height == 30) {
				reward += maintainBonus;
			} else if (ar_state.ticks_at_optimal_height == 60) {
				reward += maintainBonus * 1.5f;
			} else if (ar_state.ticks_at_optimal_height == 120) {
				reward += maintainBonus * 2.0f;
			}
		}
		else if (height_diff <= maxHeightDeviation) {
			// Outside tolerance but not too far - reduced reward
			float deviation_ratio = (height_diff - optimalHeightTolerance) / 
			                        (maxHeightDeviation - optimalHeightTolerance);
			float height_quality = 1.0f - deviation_ratio;
			reward += heightMaintenanceWeight * height_quality * (1.0f - heightPenaltyScale);
			ar_state.ticks_at_optimal_height = 0;
		}
		else {
			// Too far from optimal - penalty
			reward -= heightMaintenanceWeight * heightPenaltyScale;
			ar_state.ticks_at_optimal_height = 0;
		}
		
		// ========================================
		// Z VELOCITY MAINTENANCE REWARD
		// ========================================
		float z_vel_diff = std::abs(ar_state.smoothed_z_vel - targetZVel);
		
		if (z_vel_diff <= zVelTolerance) {
			// Good Z velocity
			float z_vel_quality = 1.0f - (z_vel_diff / zVelTolerance);
			reward += zVelMaintenanceWeight * z_vel_quality;
		}
		else {
			// Z velocity too different from target
			float excess = z_vel_diff - zVelTolerance;
			float penalty = std::min(excess / 500.0f, 1.0f) * zVelMaintenanceWeight * 0.5f;
			reward -= penalty;
		}
		
		// ========================================
		// RISING TOWARD OPTIMAL BONUS
		// ========================================
		if (car_height < target_height - optimalHeightTolerance) {
			// Below optimal - reward rising
			if (current_z_vel > 100.0f) {
				float rising_quality = clamp01(current_z_vel / 800.0f);
				reward += risingBonus * rising_quality;
			}
		}
		else if (car_height > target_height + optimalHeightTolerance) {
			// Above optimal - reward controlled descent (not crashing)
			if (current_z_vel > -300.0f && current_z_vel < 100.0f) {
				reward += risingBonus * 0.5f;  // Controlled hover/slight descent
			}
		}
		
		ar_state.prev_height = car_height;
		return reward;
	}

	float GetFieldPositionMultiplier(const Player& player, const Vec& ball_pos) {
		if (!use_field_scaling) return 1.0f;

		Vec oppGoal = (player.team == RocketSim::Team::BLUE)
			? CommonValues::ORANGE_GOAL_CENTER
			: CommonValues::BLUE_GOAL_CENTER;
		Vec ownGoal = (player.team == RocketSim::Team::BLUE)
			? CommonValues::BLUE_GOAL_CENTER
			: CommonValues::ORANGE_GOAL_CENTER;

		float dist_to_opp_goal = (ball_pos - oppGoal).Length();
		float field_mult = 1.0f;

		if (dist_to_opp_goal <= optimal_start_dist) {
			float proximity_factor = 1.0f - (dist_to_opp_goal / optimal_start_dist);
			field_mult = 1.0f + (max_field_mult - 1.0f) * proximity_factor;
		}
		else if (dist_to_opp_goal <= too_far_penalty_dist) {
			float penalty_factor = (dist_to_opp_goal - optimal_start_dist) /
				(too_far_penalty_dist - optimal_start_dist);
			field_mult = 1.0f - (1.0f - min_field_mult) * penalty_factor;
		}
		else {
			field_mult = min_field_mult;
		}

		float midfield_y = (oppGoal.y + ownGoal.y) / 2.0f;
		bool in_own_half = (player.team == RocketSim::Team::BLUE && ball_pos.y < midfield_y) ||
			(player.team == RocketSim::Team::ORANGE && ball_pos.y > midfield_y);
		if (in_own_half) {
			field_mult *= 0.85f;
		}

		return std::max(min_field_mult, std::min(max_field_mult, field_mult));
	}

	float GetShotQuality(const Vec& player_pos, const Vec& ball_pos,
		const Vec& ball_to_goal_dir, float ball_height) {
		Vec player_to_ball = ball_pos - player_pos;
		player_to_ball.z = 0.0f;
		float dist = player_to_ball.Length();

		if (dist < 1e-6f) return 0.5f;

		Vec approach_dir = player_to_ball / dist;
		float dot = approach_dir.Dot(ball_to_goal_dir);
		dot = std::max(-1.0f, std::min(1.0f, dot));

		float angle_rad = std::acos(dot);
		float angle_deg = angle_rad * 180.0f / 3.14159265f;

		float angle_quality = 0.0f;
		if (angle_deg < optimal_approach_angle) {
			angle_quality = 1.0f;
		}
		else if (angle_deg < optimal_approach_angle + 40.0f) {
			float ratio = (angle_deg - optimal_approach_angle) / 40.0f;
			angle_quality = 1.0f - 0.6f * ratio;
		}
		else {
			angle_quality = 0.4f;
		}

		float height_diff = std::abs(ball_height - optimal_shot_height);
		float height_quality = 1.0f;
		if (height_diff > 400.0f) {
			height_quality = 0.5f;
		}
		else {
			height_quality = 1.0f - 0.5f * (height_diff / 400.0f);
		}

		return angle_quality * height_quality;
	}

	// ============================================================================
	// MAIN REWARD FUNCTION
	// ============================================================================

	float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
		auto& ar_state = player_states[player.carId];

		// ========================================================================
		// GATING
		// ========================================================================
		bool passed_gates = true;
		if (player.isOnGround || player.pos.z < min_air_height) {
			passed_gates = false;
		}

		float dist = (state.ball.pos - player.pos).Length();
		if (dist > max_ball_dist) {
			passed_gates = false;
		}

		if (!passed_gates) {
			ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - decay_rate_gating);
			ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_gating);
			ar_state.smoothed_roll_rate *= 0.4f;
			ar_state.has_reached_min_ticks = false;
			ar_state.best_shot_quality = 0.0f;
			ar_state.ticks_in_good_position = 0;
			ar_state.ticks_at_optimal_height = 0;
			ar_state.consecutive_touches = 0;
			return 0.0f;
		}

		float field_mult = GetFieldPositionMultiplier(player, state.ball.pos);

		// ========================================================================
		// 🆕 HEIGHT MAINTENANCE REWARD
		// ========================================================================
		float height_reward = GetHeightMaintenanceReward(player, state.ball.pos.z, ar_state);
		float reward = height_reward * field_mult;

		// ========================================================================
		// 🆕 PROXIMITY REWARD (staying close to ball)
		// ========================================================================
		if (dist < maxProximityDistance) {
			float proximity_quality = 1.0f - (dist / maxProximityDistance);
			reward += proximityWeight * proximity_quality * field_mult;
		}

		// ========================================================================
		// DIREZIONI BASE
		// ========================================================================
		Vec toBall = state.ball.pos - player.pos;
		Vec toBall2D(toBall.x, toBall.y, 0.0f);
		float d2 = toBall2D.Length();
		Vec dirToBall2D = (d2 > 1e-6f) ? (toBall2D / d2) : Vec(0, 0, 0);

		Vec v2D(player.vel.x, player.vel.y, 0.0f);
		float v2 = v2D.Length();
		Vec vHat2D = (v2 > 1e-6f) ? (v2D / v2) : Vec(0, 0, 0);

		// ========================================================================
		// DIREZIONE TARGET BLEND
		// ========================================================================
		Vec oppGoal = (player.team == RocketSim::Team::BLUE)
			? CommonValues::ORANGE_GOAL_CENTER
			: CommonValues::BLUE_GOAL_CENTER;

		Vec ballToGoal2D(oppGoal.x - state.ball.pos.x, oppGoal.y - state.ball.pos.y, 0.0f);
		float bgL = ballToGoal2D.Length();
		Vec dirBallToGoal2D = (bgL > 1e-6f) ? (ballToGoal2D / bgL) : Vec(0, 0, 0);

		float nearBall01 = clamp01((max_ball_dist - std::min(max_ball_dist, d2)) / max_ball_dist);
		float wB = w_dir_ball * (0.6f + 0.4f * nearBall01);
		float wG = w_dir_goal * (1.0f - 0.3f * nearBall01);

		Vec desired2D = wB * dirToBall2D + wG * dirBallToGoal2D;
		float desL = desired2D.Length();
		if (desL > 1e-6f) desired2D = desired2D / desL;

		Vec fwd2D(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
		float f2 = fwd2D.Length();
		if (f2 > 1e-6f) fwd2D = fwd2D / f2;

		// Velocity/forward dot towards goal (ball-to-goal direction)
		float vel_dot_goal = vHat2D.Dot(dirBallToGoal2D);
		float fwd_dot_goal = fwd2D.Dot(dirBallToGoal2D);
		bool towards_net = (vel_dot_goal > min_goal_push_dot);

		// Ball alignment: always reward (going towards ball)
		float align_ball_vel = clamp01(vHat2D.Dot(dirToBall2D));
		float align_ball_fwd = clamp01(fwd2D.Dot(dirToBall2D));

		// Goal alignment: only reward when moving towards the net
		float align_goal_vel = towards_net ? clamp01(vel_dot_goal) : 0.0f;
		float align_goal_fwd = towards_net ? clamp01(fwd_dot_goal) : 0.0f;

		float align_vel_desired = clamp01(vHat2D.Dot(desired2D));
		float cross = std::sqrt(std::max(0.0f, 1.0f - align_vel_desired * align_vel_desired));
		float cross_pen = cross_track_penalty * cross;

		float opposite_pen = 0.0f;
		if (v2 > 50.0f) {
			float rawDot = vHat2D.Dot(desired2D);
			if (rawDot < 0.0f) {
				opposite_pen = opposite_dir_penalty * (-rawDot);
			}
		}

		// ========================================================================
		// ALLINEAMENTI CLASSICI
		// ========================================================================
		float vlen = player.vel.Length();
		Vec vHat = (vlen > 1e-6f) ? (player.vel / vlen) : Vec(0, 0, 0);
		Vec dirToBall = (dist > 1e-6f) ? ((state.ball.pos - player.pos) / dist) : Vec(0, 0, 0);
		float align_ball = clamp01(vHat.Dot(dirToBall));

		const float R = CommonValues::BALL_RADIUS;
		float proximity = clamp01(1.0f - (dist - R) / std::max(1.0f, (max_ball_dist - R)));

		// ========================================================================
		// REWARD BASE PER-TICK (goal part only when moving towards net)
		// ========================================================================
		float shaped_core =
			0.25f * align_ball_vel +
			0.15f * align_ball_fwd +
			0.35f * align_goal_vel +
			0.15f * align_goal_fwd +
			0.10f * align_ball +
			0.10f * proximity;

		shaped_core = clamp01(shaped_core);
		reward += per_tick_scale * shaped_core * field_mult;

		reward -= (cross_pen + opposite_pen) * field_mult;

		// ========================================================================
		// SHOT QUALITY BONUS (only when moving towards net)
		// ========================================================================
		if (towards_net && dist < max_ball_dist * shot_quality_distance_max) {
			float shot_quality = GetShotQuality(player.pos, state.ball.pos,
				dirBallToGoal2D, state.ball.pos.z);

			reward += shot_quality_bonus * shot_quality * field_mult * 0.04f;

			ar_state.best_shot_quality = std::max(ar_state.best_shot_quality, shot_quality);

			if (shot_quality > 0.75f) {
				ar_state.ticks_in_good_position++;

				if (ar_state.ticks_in_good_position == 15) {
					reward += 2.0f * field_mult;
				}
				else if (ar_state.ticks_in_good_position == 30) {
					reward += 3.0f * field_mult;
				}
			}
			else {
				ar_state.ticks_in_good_position = 0;
			}

			float height_diff = std::abs(state.ball.pos.z - optimal_shot_height);
			if (height_diff < 150.0f) {
				float height_quality = 1.0f - (height_diff / 150.0f);
				reward += shot_height_bonus * height_quality * field_mult * 0.03f;
			}
		}
		else if (!towards_net) {
			ar_state.ticks_in_good_position = 0;
		}

		// ========================================================================
		// TOUCH BONUS
		// ========================================================================
		if (player.ballTouchedStep) {
			if (towards_net)
				ar_state.consecutive_touches = std::min(ar_state.consecutive_touches + 1, maxConsecutiveTouches);
			else
				ar_state.consecutive_touches = 0;
			
			Vec dV = (use_impulse_delta && state.prev)
				? (state.ball.vel - state.prev->ball.vel)
				: state.ball.vel;

			Vec dV2D(dV.x, dV.y, 0.0f);
			float dVlen = dV2D.Length();

			if (dVlen > 1e-6f && bgL > 1e-6f) {
				Vec dVhat = dV2D / dVlen;
				float dot_goal = std::max(-1.0f, std::min(1.0f, dVhat.Dot(dirBallToGoal2D)));

				float e_pos = std::exp(lambda_goal * dot_goal);
				float e_min = std::exp(-lambda_goal);
				float e_max = std::exp(lambda_goal);
				float gain01 = (e_pos - e_min) / (e_max - e_min);

				float mag = clamp01(dVlen / std::max(1.0f, max_delta_v));

				// Goal-directed touch bonus: only when ball goes towards net AND player was moving towards net
				float dir_factor = (dot_goal > 0.0f && towards_net) ? gain01 : 0.0f;

				float touch = touch_base * mag * dir_factor * field_mult;

				// 🆕 Consecutive touch bonus (air dribble)
				if (ar_state.consecutive_touches > 1) {
					float touch_streak_bonus = 1.0f + 0.1f * std::min(ar_state.consecutive_touches - 1, 10);
					touch *= touch_streak_bonus;
				}

				if (towards_net && dot_goal > 0.5f) {
					float vertical_gap = state.ball.pos.z - player.pos.z;
					if (vertical_gap > under_ball_min_gap && dV.z > 0.0f) {
						float gap01 = clamp01((vertical_gap - under_ball_min_gap) /
							std::max(1.0f, under_ball_max_gap - under_ball_min_gap));
						float touch_boost = 1.0f + under_touch_boost * gap01;
						touch *= touch_boost;
					}
				}

				reward += touch;

				if (towards_net && dot_goal > 0.9f && mag > 0.6f) {
					reward += 5.0f * field_mult;
				}
			}
		} else {
			// Decay consecutive touches if no touch this tick
			// (keep it if still in air and close to ball)
			if (dist > maxProximityDistance * 1.5f) {
				ar_state.consecutive_touches = 0;
			}
		}

		// ========================================================================
		// AIR ROLL BONUS (only when moving towards net)
		// ========================================================================
		if (vlen > airroll_speed_threshold && towards_net) {
			Vec forward = player.rotMat.forward;
			float roll_rate_signed = player.angVel.Dot(forward);
			float roll_rate_abs = std::fabs(roll_rate_signed);

			float smooth_factor = (roll_rate_abs > min_airroll_rate)
				? smooth_factor_active
				: smooth_factor_inactive;
			ar_state.smoothed_roll_rate = smooth_factor * ar_state.smoothed_roll_rate +
				(1.0f - smooth_factor) * roll_rate_abs;

			float direction_threshold = min_airroll_rate * direction_detect_mult;
			float curr_direction = 0.0f;
			if (roll_rate_signed > direction_threshold) curr_direction = 1.0f;
			if (roll_rate_signed < -direction_threshold) curr_direction = -1.0f;

			bool direction_changed = false;
			if (curr_direction != 0.0f && ar_state.prev_roll_direction != 0.0f) {
				if (curr_direction * ar_state.prev_roll_direction < 0) {
					direction_changed = true;
				}
			}

			if (ar_state.smoothed_roll_rate > min_airroll_rate && !direction_changed) {
				ar_state.consecutive_ticks++;
				ar_state.stable_direction_ticks++;
				ar_state.total_airroll_time += 1.0f;

				if (ar_state.stable_direction_ticks >= min_ticks_for_bonus) {
					ar_state.has_reached_min_ticks = true;
				}

				if (ar_state.has_reached_min_ticks) {
					float progression = std::min(
						static_cast<float>(ar_state.stable_direction_ticks) /
						static_cast<float>(max_progression_ticks),
						1.0f
					);

					float current_bonus = airroll_bonus_base +
						(airroll_bonus_max - airroll_bonus_base) * progression;

					float intensity_factor = clamp01(ar_state.smoothed_roll_rate / 1.0f);
					intensity_factor = std::max(0.5f, intensity_factor);

					float speed_factor = clamp01(vlen / 1800.0f);
					speed_factor = 0.6f + 0.4f * speed_factor;

					float proximity_factor = clamp01(proximity * 2.0f);
					proximity_factor = 0.7f + 0.3f * proximity_factor;

					float consistency_factor = 1.0f;
					if (ar_state.stable_direction_ticks > consistency_start_ticks) {
						float consistency_progress = std::min(
							static_cast<float>(ar_state.stable_direction_ticks - consistency_start_ticks) / 80.0f,
							1.0f
						);
						consistency_factor = 1.0f + consistency_bonus_rate * consistency_progress;
					}

					float desired_factor = 0.7f + 0.3f * align_vel_desired;

					float avg_quality = (intensity_factor + speed_factor + proximity_factor) / 3.0f;
					float quality_mult = (avg_quality > quality_threshold) ? quality_multiplier : 1.0f;

					float airroll_bonus = current_bonus *
						intensity_factor *
						speed_factor *
						proximity_factor *
						consistency_factor *
						desired_factor *
						quality_mult *
						field_mult;

					ar_state.accumulated_bonus += airroll_bonus;
					reward += airroll_bonus;

					if (ar_state.stable_direction_ticks == 60 ||
						ar_state.stable_direction_ticks == 120) {
						reward += 0.5f * field_mult;
					}
				}
			}
			else {
				if (direction_changed) {
					ar_state.consecutive_ticks = 0;
					ar_state.stable_direction_ticks = 0;
					ar_state.accumulated_bonus = 0.0f;
					ar_state.has_reached_min_ticks = false;
				}
				else if (ar_state.smoothed_roll_rate <= min_airroll_rate) {
					ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - decay_rate_active);
					ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_active);

					if (ar_state.stable_direction_ticks < min_ticks_for_bonus) {
						ar_state.has_reached_min_ticks = false;
					}
				}
			}

			if (curr_direction != 0.0f) {
				ar_state.prev_roll_direction = curr_direction;
			}

		}
		else {
			ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - 1);
			ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_slow);
			ar_state.smoothed_roll_rate *= 0.75f;

			if (ar_state.stable_direction_ticks < min_ticks_for_bonus) {
				ar_state.has_reached_min_ticks = false;
			}
		}

		return reward;
	}
};





class TeammateBumpPenaltyReward : public Reward {
	private:
		std::unordered_map<int, uint32_t> previousContactCarID;
		std::unordered_map<int, float> previousContactTimer;

	public:
		TeammateBumpPenaltyReward() {}

		virtual void Reset(const GameState& initialState) override {
			previousContactCarID.clear();
			previousContactTimer.clear();
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float reward = 0.0f;

			uint32_t currentContactCarID = player.carContact.otherCarID;
			float currentContactTimer = player.carContact.cooldownTimer;

			auto prevCarIDIter = previousContactCarID.find(player.carId);
			auto prevTimerIter = previousContactTimer.find(player.carId);

			bool hadPreviousState = (prevCarIDIter != previousContactCarID.end() &&
				prevTimerIter != previousContactTimer.end());

			if (currentContactTimer > 0 && currentContactCarID != 0) {
				bool isNewBump = false;

				if (!hadPreviousState) {
					isNewBump = true;
				}
				else {
					float prevTimer = prevTimerIter->second;
					uint32_t prevCarID = prevCarIDIter->second;

					isNewBump = (currentContactTimer > prevTimer) ||
						(currentContactCarID != prevCarID && prevTimer <= 0);
				}

				if (isNewBump) {
					for (const auto& otherPlayer : state.players) {
						if (otherPlayer.carId == currentContactCarID) {
							if (otherPlayer.team == player.team) {
								reward -= 1.0f;
							}
							break;
						}
					}
				}
			}

			previousContactCarID[player.carId] = currentContactCarID;
			previousContactTimer[player.carId] = currentContactTimer;

			return reward;
		}
	};



	class KaiyoEnergyReward : public Reward {
	public:
		const double GRAVITY = 650;
		const double MASS = 180;
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			const auto max_energy = (MASS * GRAVITY * (CommonValues::CEILING_Z - 17.)) + (0.5 * MASS * (CommonValues::CAR_MAX_SPEED * CommonValues::CAR_MAX_SPEED));
			double energy = 0;

			if (player.HasFlipOrJump()) {
				energy += 0.35 * MASS * 292. * 292.;
			}

			if (player.HasFlipOrJump() && !player.isOnGround) {
				energy += 0.35 * MASS * 550. * 550.;
			}

			energy += MASS * GRAVITY * (player.pos.z - 17.) * 0.75;

			double velocity = player.vel.Length();
			energy += 0.5 * MASS * velocity * velocity;
			energy += 7.97e6 * player.boost;

			double norm_energy = player.isDemoed ? 0.0f : (energy / max_energy);

			return static_cast<float>(norm_energy);
		}
	};

	class HyperNoStackReward : public Reward {
	private:
		struct PlayerState {
			bool wasOnWall = false;
			bool dashUsed = false;
			bool hadFlipBefore = false;
			bool wasFlipping = false;
			Vec prevVel = Vec(0, 0, 0);
			// int lastDemoCount = 0; // Removed: Not supported by current Player struct
		};

		std::map<uint32_t, PlayerState> playerStates;

		// --- Configuration ---
		float wallHeightThreshold;

		// Base Rewards
		float dashRewardBase;
		float resetRewardBase;
		float wavedashRewardBase;
		float zapDashBaseReward;
		// float demoReward; // Removed temporarily to fix build

		// Quality Scaling
		float accelerationScalar;

		// Penalties
		float wallStayPenalty;
		float supersonicBoostPenalty;

		// Zap Dash Config
		float zapMinSpeedGain;
		float zapMinNoseDown;
		float zapMinFwdDot;

		bool debug;

	public:
		HyperNoStackReward(
			// Base Values
			float dashRewardBase = 12.0f,
			float resetRewardBase = 16.0f,
			float wavedashRewardBase = 0.0f,
			float zapDashBaseReward = 15.0f,
			// float demoReward = 30.0f, 
			// Scaling
			float accelerationScalar = 1.0f,
			// Penalties
			float wallStayPenalty = 0.5f,
			float supersonicBoostPenalty = -0.1f,
			// Debug
			bool debug = false
		) :
			wallHeightThreshold(100.0f),
			dashRewardBase(dashRewardBase), resetRewardBase(resetRewardBase),
			wavedashRewardBase(wavedashRewardBase), zapDashBaseReward(zapDashBaseReward),
			// demoReward(demoReward),
			accelerationScalar(accelerationScalar),
			wallStayPenalty(wallStayPenalty), supersonicBoostPenalty(supersonicBoostPenalty),
			zapMinSpeedGain(500.0f), zapMinNoseDown(0.5f), zapMinFwdDot(0.7f),
			debug(debug)
		{
		}

		virtual void Reset(const GameState& initialState) override {
			playerStates.clear();
			for (const auto& player : initialState.players) {
				InitState(player);
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev) return 0.0f;

			if (playerStates.find(player.carId) == playerStates.end()) {
				InitState(player);
			}
			PlayerState& st = playerStates[player.carId];

			float totalReward = 0.0f;
			float dt = 1.0f / 120.0f;

			// --- PRE-CALCULATIONS ---
			bool isOnWall = IsOnWall(player.pos);
			bool hasFlip = player.HasFlipOrJump();

			// Calculate Acceleration Quality
			float acceleration = (player.vel - st.prevVel).Length() / dt;
			float normAccel = std::min(acceleration / 5000.0f, 2.0f);
			float qualityMult = 1.0f + (normAccel * accelerationScalar);

			// =========================================================
			// MECHANIC DETECTION
			// =========================================================

			// 1. DEMOLITION (Disabled to fix build error)
			/*
			if (player.match_demolitions > st.lastDemoCount) {
				totalReward += demoReward * qualityMult;
				if (debug) printf("ID:%d | DEMO | Reward: %.2f\n", player.carId, demoReward * qualityMult);
			}
			*/

			// 2. ZAP DASH
			float zapRatio = CheckZapDash(player);
			if (zapRatio > 0.0f) {
				float r = zapDashBaseReward * zapRatio * qualityMult;
				totalReward += r;
				if (debug) printf("ID:%d | ZapDash | Reward: %.2f\n", player.carId, r);
			}

			// 3. WALL DASH (LAUNCH)
			if (st.wasOnWall && st.hadFlipBefore && !hasFlip && !isOnWall) {
				st.dashUsed = true;
				float r = dashRewardBase * qualityMult;
				totalReward += r;
				if (debug) printf("ID:%d | WallDash_Launch | Reward: %.2f\n", player.carId, r);
			}

			// 4. WALL DASH (RESET)
			if (st.dashUsed && isOnWall && hasFlip) {
				st.dashUsed = false; // Cycle complete
				float r = resetRewardBase * qualityMult;
				totalReward += r;
				if (debug) printf("ID:%d | WallDash_Reset | Reward: %.2f\n", player.carId, r);
			}

			// 5. WAVEDASH
			const Player* prevPlayer = nullptr;
			for (const auto& p : state.prev->players) { if (p.carId == player.carId) { prevPlayer = &p; break; } }

			bool justLanded = player.isOnGround && prevPlayer && !prevPlayer->isOnGround;
			if (justLanded && (player.isFlipping || st.wasFlipping)) {
				float r = wavedashRewardBase * qualityMult;
				totalReward += r;
				if (debug) printf("ID:%d | Wavedash | Reward: %.2f\n", player.carId, r);
			}

			// =========================================================
			// PENALTIES
			// =========================================================

			// 1. Supersonic Boost Waste
			if (prevPlayer && player.isSupersonic && player.boost < prevPlayer->boost) {
				totalReward += supersonicBoostPenalty;
			}

			// 2. Conditional Wall Stay Penalty
			if (isOnWall && st.dashUsed && player.vel.Length() < 600.0f) {
				totalReward += wallStayPenalty;
			}

			// =========================================================
			// STATE UPDATES
			// =========================================================
			st.wasOnWall = isOnWall;
			st.hadFlipBefore = hasFlip;
			st.wasFlipping = player.isFlipping;
			st.prevVel = player.vel;
			// st.lastDemoCount = player.match_demolitions; // Disabled

			if (player.isOnGround) {
				st.dashUsed = false;
			}

			return totalReward;
		}

	private:
		void InitState(const Player& p) {
			PlayerState newState;
			newState.prevVel = p.vel;
			newState.hadFlipBefore = p.HasFlipOrJump();
			// newState.lastDemoCount = p.match_demolitions; // Disabled
			playerStates[p.carId] = newState;
		}

		float CheckZapDash(const Player& current) {
			if (!current.prev || !current.prev->prev || !current.prev->prev->prev) return 0.0f;

			const Player* p0 = &current;
			const Player* p1 = current.prev;
			const Player* p2 = p1->prev;
			const Player* p3 = p2->prev;

			bool wasAirborneT3 = !p3->isOnGround;
			bool landedT2 = wasAirborneT3 && p2->isOnGround;
			bool noseDown = p2->rotMat.forward.z < -zapMinNoseDown;
			bool wasMovingDown = p2->vel.z < -200.0f;
			bool executingFlip = p0->isFlipping;

			if (landedT2 && noseDown && wasMovingDown && executingFlip) {
				Vec carFwd = p1->rotMat.forward;
				Vec velDir = p0->vel.Normalized();
				if (carFwd.Dot(velDir) > zapMinFwdDot) {
					float gain = p0->vel.Length() - p2->vel.Length();
					if (gain > zapMinSpeedGain) {
						float bonus = (p0->vel.Length() > 2200.f) ? 1.5f : 1.0f;
						return (gain / 1000.0f) * bonus;
					}
				}
			}
			return 0.0f;
		}

		bool IsOnWall(const Vec& pos) {
			if (pos.z < wallHeightThreshold) return false;
			if (std::abs(std::abs(pos.x) - 4096.0f) < 150.0f) return true;
			if (std::abs(std::abs(pos.y) - 5120.0f) < 150.0f) return true;
			return false;
		}
	};

	class FlickReward45Degree : public Reward {
	private:
		struct PlayerState {
			bool wasOnGround;
			bool ballWasOnRoof;
			float timeSinceBallOnRoof;
			Vec lastCarPos;

			PlayerState() : wasOnGround(true), ballWasOnRoof(false),
				timeSinceBallOnRoof(0.0f), lastCarPos(0, 0, 0) {
			}
		};

		static constexpr float TICK_SKIP = 4.0f;
		static constexpr float DT = (1.0f / 120.0f) * TICK_SKIP;
		std::unordered_map<uint32_t, PlayerState> playerStates;

	public:
		float minBallSpeed;
		float maxBallSpeed;
		float angleTolerance;
		float roofDistanceMax;
		float maxTimeSinceRoof;

		FlickReward45Degree(
			float minBallSpeed = 1200.0f,
			float maxBallSpeed = 3500.0f,
			float angleTolerance = 20.0f,
			float roofDistanceMax = 150.0f,
			float maxTimeSinceRoof = 0.3f
		) : minBallSpeed(minBallSpeed), maxBallSpeed(maxBallSpeed),
			angleTolerance(angleTolerance), roofDistanceMax(roofDistanceMax),
			maxTimeSinceRoof(maxTimeSinceRoof) {
		}

		virtual void Reset(const GameState& initialState) override {
			playerStates.clear();
			for (const auto& player : initialState.players) {
				PlayerState state;
				state.wasOnGround = player.isOnGround;
				state.ballWasOnRoof = false;
				state.timeSinceBallOnRoof = 999.0f;
				state.lastCarPos = player.pos;
				playerStates[player.carId] = state;
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			uint32_t carId = player.carId;

			// Initialize if needed
			if (playerStates.find(carId) == playerStates.end()) {
				PlayerState newState;
				newState.wasOnGround = player.isOnGround;
				newState.ballWasOnRoof = false;
				newState.timeSinceBallOnRoof = 999.0f;
				newState.lastCarPos = player.pos;
				playerStates[carId] = newState;
			}

			PlayerState& ps = playerStates[carId];

			// Check if ball is currently on car roof
			Vec carToBall = state.ball.pos - player.pos;
			float distToBall = carToBall.Length();
			Vec carUp = player.rotMat.up;

			// Ball is "on roof" if it's close, above the car, and aligned with car's up vector
			bool ballOnRoofNow = false;
			if (distToBall < roofDistanceMax) {
				float heightAboveCar = carToBall.Dot(carUp);
				if (heightAboveCar > 50.0f && heightAboveCar < 150.0f) {
					// Check if ball is roughly above car (not to the side)
					Vec carToBallHorizontal = carToBall - (carUp * heightAboveCar);
					if (carToBallHorizontal.Length() < 80.0f) {
						ballOnRoofNow = true;
					}
				}
			}

			// Update timing
			if (ballOnRoofNow) {
				ps.timeSinceBallOnRoof = 0.0f;
				ps.ballWasOnRoof = true;
			}
			else {
				ps.timeSinceBallOnRoof += DT;
			}

			float reward = 0.0f;

			// Detect flick: player touched ball recently after having it on roof
			if (player.ballTouchedStep && ps.ballWasOnRoof && ps.timeSinceBallOnRoof < maxTimeSinceRoof) {

				// Check if player just left ground (jumped for flick)
				bool justJumped = ps.wasOnGround && !player.isOnGround;

				if (!state.prev) {
					ps.wasOnGround = player.isOnGround;
					ps.lastCarPos = player.pos;
					return 0.0f;
				}

				// Check ball velocity for 45-degree angle
				float ballSpeed = state.ball.vel.Length();

				if (ballSpeed >= minBallSpeed) {
					Vec ballVelNorm = state.ball.vel.Normalized();

					// Calculate angle from horizontal
					float horizontalSpeed = sqrtf(ballVelNorm.x * ballVelNorm.x + ballVelNorm.y * ballVelNorm.y);
					float angleDeg = atan2f(ballVelNorm.z, horizontalSpeed) * 57.2957795f;

					// Check if angle is near 45 degrees
					float angleDiff = fabsf(angleDeg - 45.0f);

					if (angleDiff <= angleTolerance) {
						// Power bonus (speed-based)
						float powerBonus = RS_MIN(1.0f, ballSpeed / maxBallSpeed);

						// Angle precision bonus
						float anglePrecision = 1.0f - (angleDiff / angleTolerance);

						// Height bonus (higher flicks are better)
						float heightBonus = RS_MIN(1.0f, state.ball.pos.z / 500.0f);

						// Jump bonus (if player jumped for the flick)
						float jumpBonus = justJumped ? 1.5f : 1.0f;

						// Timing bonus (quicker flick after roof = better)
						float timingBonus = 1.0f - (ps.timeSinceBallOnRoof / maxTimeSinceRoof);
						timingBonus = RS_MAX(0.3f, timingBonus);

						reward = powerBonus * anglePrecision * heightBonus * jumpBonus * timingBonus;

						// Reset state after successful flick
						ps.ballWasOnRoof = false;
						ps.timeSinceBallOnRoof = 999.0f;
					}
				}
			}

			// Update state for next frame (ALWAYS, not just when reward is 0!)
			ps.wasOnGround = player.isOnGround;
			ps.lastCarPos = player.pos;

			return reward;
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/ball_goal_rewards.py
	class VelocityBallToGoalReward : public Reward {
	public:
		bool ownGoal = false;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			bool targetOrangeGoal = player.team == Team::BLUE;
			if (ownGoal) targetOrangeGoal = !targetOrangeGoal;

			Vec targetPos = targetOrangeGoal
				? CommonValues::ORANGE_GOAL_BACK
				: CommonValues::BLUE_GOAL_BACK;

			Vec ballDirToGoal = (targetPos - state.ball.pos).Normalized();
			return ballDirToGoal.Dot(state.ball.vel / CommonValues::BALL_MAX_SPEED);
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class VelocityPlayerToBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			Vec normVel = player.vel / CommonValues::CAR_MAX_SPEED;
			return dirToBall.Dot(normVel);
		}
	};

	class AirdribbleSetupReward : public Reward {
	public:
		// Tunable parameters for the reward function
		const float MIN_BALL_HEIGHT = CommonValues::BALL_RADIUS + 15.f;
		const float MIN_UPWARD_VELOCITY = 100.f;
		const float MAX_DISTANCE_TO_BALL = 500.f;
		const float IDEAL_HEIGHT_DIFFERENCE = 150.f; // Bot should be this far below the ball
		const float MAX_REWARDED_POP_VEL = 1500.f;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// We need the previous state to analyze the result of the touch.
			if (!player.prev || !player.ballTouchedStep) {
				return 0.f;
			}

			const auto& prevBallState = state.prev->ball;
			const auto& ballState = state.ball;
			const auto& playerState = player;

			// 1. The Pop-up: Ensure the ball is popped into the air with upward velocity.
			if (ballState.pos.z < MIN_BALL_HEIGHT || ballState.vel.z < MIN_UPWARD_VELOCITY) {
				return 0.f;
			}

			// Reward for imparting upward velocity on the ball.
			float popReward = RS_CLAMP(ballState.vel.z / MAX_REWARDED_POP_VEL, 0.f, 1.f);

			// 2. Player Follow-up: Ensure the player is also airborne and following the ball.
			if (playerState.isOnGround) {
				return 0.f;
			}

			// Reward for being close to the ball.
			float distToBall = playerState.pos.Dist(ballState.pos);
			if (distToBall > MAX_DISTANCE_TO_BALL) {
				return 0.f;
			}
			float proximityReward = 1.f - (distToBall / MAX_DISTANCE_TO_BALL);

			// Reward for having velocity directed towards the ball.
			Vec dirToBall = (ballState.pos - playerState.pos).Normalized();
			float velocityAlignmentReward = (playerState.vel.Normalized().Dot(dirToBall) + 1.f) / 2.f;

			// 3. Positioning: Reward for being underneath the ball.
			if (playerState.pos.z >= ballState.pos.z) {
				return 0.f;
			}

			// Use a Gaussian function to give max reward at an ideal height difference.
			float heightDiff = ballState.pos.z - playerState.pos.z;
			float underBallReward = exp(-pow(heightDiff - IDEAL_HEIGHT_DIFFERENCE, 2) / (2 * pow(50, 2)));

			// 4. Goal Alignment: Reward for setting up towards the opponent's goal.
			Vec opponentGoal = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec ballToGoalDir = (opponentGoal - ballState.pos).Normalized();
			float goalAlignmentReward = (ballState.vel.Normalized().Dot(ballToGoalDir) + 1.f) / 2.f;

			// Combine all reward components.
			// By multiplying, we ensure that all conditions must be met to receive a reward.
			float totalReward =
				popReward *
				proximityReward *
				velocityAlignmentReward *
				underBallReward *
				goalAlignmentReward;

			if (isnan(totalReward) || isinf(totalReward)) {
				return 0.f;
			}

			return totalReward;
		}
	};



	class DoubleTapReward : public Reward {
	private:
		// Enum to track the state of a double tap attempt
		enum class AttemptState {
			IDLE,                   // Not currently attempting a double tap
			AWAITING_BACKBOARD_HIT, // Player has hit the ball towards the backboard
			AWAITING_SECOND_TOUCH,  // Ball has hit the backboard, waiting for player's follow-up
			AWAITING_GOAL           // Player has made the follow-up touch, waiting for a goal
		};

		// Struct to hold the state for each player's attempt
		struct DoubleTapAttempt {
			AttemptState state = AttemptState::IDLE;
			uint64_t last_update_tick = 0;
			uint64_t floor_hit_tick = 0;
		};

		std::vector<DoubleTapAttempt> player_attempts;
		bool debug;

	public:
		/**
		 * @brief Construct a new Double Tap Reward object.
		 * @param debug_mode If true, will print messages to the console for tracking double tap events.
		 */
		DoubleTapReward(bool debug_mode = false) : debug(debug_mode) {}

		// Called once when the environment is reset.
		virtual void Reset(const GameState& initialState) override {
			player_attempts.clear();
			player_attempts.resize(initialState.players.size());
		}

		// Called for each player every step to calculate their reward.
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev) return 0; // Need previous state for comparisons

			auto& attempt = player_attempts[player.index];

			// This is a safe way to get tick_time, even with tick skip.
			float tick_time = (state.lastTickCount > state.prev->lastTickCount) ?
				(state.deltaTime / (state.lastTickCount - state.prev->lastTickCount)) :
				(1.f / 120.f);

			// Timeout the attempt if too much time has passed (e.g., 5 seconds)
			if (attempt.state != AttemptState::IDLE && (state.lastTickCount - attempt.last_update_tick) * tick_time > 5.0f) {
				attempt = {}; // Reset state
			}

			// Invalidate attempt if another player touches the ball
			if (state.lastTouchCarID != -1 && state.lastTouchCarID != player.carId && attempt.state != AttemptState::IDLE) {
				attempt = {}; // Reset state
			}

			// State machine for the double tap attempt
			switch (attempt.state) {
			case AttemptState::IDLE: {
				if (player.ballTouchedStep) {
					// Heuristic: Is it a potential setup touch? Ball hit high and fast towards the opponent's backboard.
					bool is_blue_team = player.team == Team::BLUE;
					float ball_y_vel = state.ball.vel.y;
					float ball_z_pos = state.ball.pos.z;

					bool heading_to_opp_backboard = (is_blue_team && ball_y_vel > 1000) || (!is_blue_team && ball_y_vel < -1000);

					if (heading_to_opp_backboard && ball_z_pos > 500) {
						attempt.state = AttemptState::AWAITING_BACKBOARD_HIT;
						attempt.last_update_tick = state.lastTickCount;
					}
				}
				break;
			}

			case AttemptState::AWAITING_BACKBOARD_HIT: {
				// Check for a backboard rebound
				float ball_y_pos = state.ball.pos.y;
				if (abs(ball_y_pos) > CommonValues::BACK_WALL_Y - CommonValues::BALL_RADIUS * 1.5) {
					float prev_ball_y_vel = state.prev->ball.vel.y;
					float cur_ball_y_vel = state.ball.vel.y;

					// Check if y-velocity has flipped sign, indicating a bounce
					if (std::signbit(prev_ball_y_vel) != std::signbit(cur_ball_y_vel)) {
						if (debug) {
							std::cout << "DEBUG: Player " << player.carId << " got a backboard rebound!" << std::endl;
						}
						attempt.state = AttemptState::AWAITING_SECOND_TOUCH;
						attempt.last_update_tick = state.lastTickCount;
					}
				}
				break;
			}

			case AttemptState::AWAITING_SECOND_TOUCH: {
				if (player.ballTouchedStep) {
					// Player made the second touch
					attempt.state = AttemptState::AWAITING_GOAL;
					attempt.last_update_tick = state.lastTickCount;
					attempt.floor_hit_tick = 0; // Reset floor hit grace period timer
				}
				break;
			}

			case AttemptState::AWAITING_GOAL: {
				// Check if ball hits the floor to start the grace period timer
				if (attempt.floor_hit_tick == 0 && state.ball.pos.z < CommonValues::BALL_RADIUS + 15 && state.prev->ball.vel.z < 0) {
					attempt.floor_hit_tick = state.lastTickCount;
				}

				if (state.goalScored) {
					bool scored_on_correct_goal = (player.team != RS_TEAM_FROM_Y(state.ball.pos.y));

					if (scored_on_correct_goal && state.lastTouchCarID == player.carId) {

						bool within_grace_period = (attempt.floor_hit_tick == 0) ||
							((state.lastTickCount - attempt.floor_hit_tick) * tick_time <= 2.0f);

						if (within_grace_period) {
							if (debug) {
								std::cout << "DEBUG: Player " << player.carId << " scored a DOUBLE TAP!" << std::endl;
							}
							attempt = {}; // Reset state for next attempt
							return 1.0f;
						}
					}
				}
				break;
			}
			}

			return 0; // No reward this step
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/player_ball_rewards.py
	class FaceBallReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			return player.rotMat.forward.Dot(dirToBall);
		}
	};

	class SpeedReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return player.vel.Length() / CommonValues::CAR_MAX_SPEED;
		}
	};

	class WavedashReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			if (!player.prev)
				return 0;

			if (player.isOnGround && (player.prev->isFlipping && !player.prev->isOnGround)) {
				return 1;
			}
			else {
				return 0;
			}
		}
	};

	class PickupBoostReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.prev)
				return 0;

			if (player.boost > player.prev->boost) {
				return sqrtf(player.boost / 100.f) - sqrtf(player.prev->boost / 100.f);
			}
			else {
				return 0;
			}
		}
	};

	// https://github.com/AechPro/rocket-league-gym-sim/blob/main/rlgym_sim/utils/reward_functions/common_rewards/misc_rewards.py
	class SaveBoostReward : public Reward {
	public:
		float exponent;
		SaveBoostReward(float exponent = 0.5f) : exponent(exponent) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			return RS_CLAMP(powf(player.boost / 100, exponent), 0, 1);
		}
	};


	class AirReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			return !player.isOnGround;
		}
	};

	// Mostly based on the classic Necto rewards
	// Total reward output for speeding the ball up to MAX_REWARDED_BALL_SPEED is 1.0
	// The bot can do this slowly (putting) or quickly (shooting)
	class TouchAccelReward : public Reward {
	public:
		constexpr static float MAX_REWARDED_BALL_SPEED = RLGC::Math::KPHToVel(110);

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float prevSpeedFrac = RS_MIN(1, state.prev->ball.vel.Length() / MAX_REWARDED_BALL_SPEED);
				float curSpeedFrac = RS_MIN(1, state.ball.vel.Length() / MAX_REWARDED_BALL_SPEED);

				if (curSpeedFrac > prevSpeedFrac) {
					return (curSpeedFrac - prevSpeedFrac);
				}
				else {
					// Not speeding up the ball so we don't care
					return 0;
				}
			}
			else {
				return 0;
			}
		}
	};

	class TeamSpacingReward : public Reward {
	public:
		float minSpacing;

		// minSpacing: Minimum distance between teammates (in Rocket League units)
		// Players get full reward when >= minSpacing apart, penalty when too close
		TeamSpacingReward(float minSpacing = 1500.0f) : minSpacing(minSpacing) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float totalReward = 0.0f;
			int teammateCount = 0;

			// Check spacing with all teammates
			for (const Player& teammate : state.players) {
				// Skip self and opponents
				if (teammate.carId == player.carId || teammate.team != player.team)
					continue;

				teammateCount++;
				float distance = (player.pos - teammate.pos).Length();

				if (distance >= minSpacing) {
					// Good spacing - full reward
					totalReward += 1.0f;
				}
				else {
					// Too close - linear penalty based on how close they are
					float ratio = distance / minSpacing;  // 0 to 1
					totalReward += ratio;  // Linear reward from 0 to 1
				}
			}

			// Return average reward across all teammates (0 if no teammates)
			return teammateCount > 0 ? totalReward / teammateCount : 0.0f;
		}
	};





    // Passive reward just for existing
    class ExistReward : public Reward {
    public:
        virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
            return 1.0f;
        }
    };



class ShieldzKORew : public Reward {
private:
    struct PlayerKickoffState {
        bool hasSpeedflipped = false;
        bool hasReached5050 = false;
        int jumpCount = 0;
        int wavedashCount = 0;
        float kickoffStartTime = 0.0f;
        bool kickoffActive = false;
        Vec lastVel = Vec(0, 0, 0);
        Vec lastAngVel = Vec(0, 0, 0);
        bool wasOnGround = true;
        float timeOffGround = 0.0f;
        float timeSinceLast5050Contact = 999.0f;
    };
    
    std::map<uint32_t, PlayerKickoffState> playerStates;
    float lastBallPosY = 0.0f;
    bool kickoffResolved = false;
    
    // Constants
    const float KICKOFF_THRESHOLD = 50.0f; // Distance from center to consider kickoff
    const float SPEEDFLIP_ROLL_THRESHOLD = 2.5f; // Angular velocity threshold
    const float SPEEDFLIP_TIME_WINDOW = 0.8f; // Time window from kickoff start
    const float BALL_CONTACT_DIST = 150.0f; // Distance to consider 50/50 contact
    const float WAVEDASH_LANDING_VEL_THRESHOLD = 100.0f; // Landing velocity for wavedash
    const float WAVEDASH_TIME_WINDOW = 0.5f; // Time after 50/50 to detect wavedash
    const float DT = 1.0f / 120.0f; // Physics tick rate
    
    bool IsKickoffPosition(const GameState& state) {
        float ballDist = sqrt(state.ball.pos.x * state.ball.pos.x + 
                             state.ball.pos.y * state.ball.pos.y);
        float ballVel = state.ball.vel.Length();
        return ballDist < KICKOFF_THRESHOLD && ballVel < 50.0f;
    }
    
    bool DetectSpeedflip(const Player& player, PlayerKickoffState& pState) {
        // Speedflip detection: High roll angular velocity + forward velocity + not on ground
        float rollRate = abs(player.angVel.z);
        bool hasHighRoll = rollRate > SPEEDFLIP_ROLL_THRESHOLD;
        bool movingForward = player.vel.Length() > 300.0f;
        bool inAir = !player.isOnGround;
        
        return hasHighRoll && movingForward && inAir;
    }
    
    bool DetectBallContact(const Player& player, const GameState& state) {
        float dist = (player.pos - state.ball.pos).Length();
        return dist < BALL_CONTACT_DIST;
    }
    
    bool DetectWavedash(const Player& player, PlayerKickoffState& pState) {
        // Wavedash: Was in air, now on ground, with significant forward velocity maintained
        bool justLanded = !pState.wasOnGround && player.isOnGround;
        bool hasSpeed = player.vel.Length() > WAVEDASH_LANDING_VEL_THRESHOLD;
        
        // Check if jump button was used (indicates dodge/cancel)
        bool usedJump = player.hasJumped;
        
        return justLanded && hasSpeed && usedJump && pState.timeOffGround > 0.1f && pState.timeOffGround < 0.4f;
    }
    
public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float totalReward = 0.0f;
        
        // Get or create player state
        if (playerStates.find(player.carId) == playerStates.end()) {
            playerStates[player.carId] = PlayerKickoffState();
        }
        PlayerKickoffState& pState = playerStates[player.carId];
        
        // Check if this is a kickoff situation
        bool isKickoff = IsKickoffPosition(state);
        
        if (isKickoff && !pState.kickoffActive) {
            // New kickoff detected - reset state
            pState = PlayerKickoffState();
            pState.kickoffActive = true;
            pState.kickoffStartTime = 0.0f;
            kickoffResolved = false;
            lastBallPosY = 0.0f;
        }
        
        if (!isKickoff && pState.kickoffActive) {
            // Kickoff ended
            pState.kickoffActive = false;
        }
        
        if (pState.kickoffActive) {
            pState.kickoffStartTime += DT;
            pState.timeSinceLast5050Contact += DT;
            
            // Update time off ground
            if (!player.isOnGround) {
                pState.timeOffGround += DT;
            } else {
                pState.timeOffGround = 0.0f;
            }
            
            // 1. Reward speedflip at kickoff start
            if (!pState.hasSpeedflipped && pState.kickoffStartTime < SPEEDFLIP_TIME_WINDOW) {
                if (DetectSpeedflip(player, pState)) {
                    pState.hasSpeedflipped = true;
                    totalReward += 1.0f; // Big reward for speedflip
                }
            }
            
            // 2. Detect 50/50 contact
            bool ballContact = DetectBallContact(player, state);
            if (ballContact && !pState.hasReached5050) {
                pState.hasReached5050 = true;
                pState.timeSinceLast5050Contact = 0.0f;
                
                // Count jumps at moment of contact
                if (player.hasJumped && !player.hasDoubleJumped) {
                    pState.jumpCount = 1;
                } else if (player.hasDoubleJumped) {
                    pState.jumpCount = 2;
                }
                
                // Reward for single jump (penalize double jump)
                if (pState.jumpCount == 1) {
                    totalReward += 0.5f; // Good - single jump
                } else if (pState.jumpCount == 2) {
                    totalReward -= 0.3f; // Bad - double jumped
                }
            }
            
            // 3. Detect wavedashes after 50/50
            if (pState.hasReached5050 && pState.timeSinceLast5050Contact < WAVEDASH_TIME_WINDOW * 3) {
                if (DetectWavedash(player, pState)) {
                    if (pState.wavedashCount == 0) {
                        // First wavedash
                        totalReward += 0.8f;
                        pState.wavedashCount++;
                    } else if (pState.wavedashCount == 1) {
                        // Second wavedash (chain)
                        totalReward += 0.6f;
                        pState.wavedashCount++;
                    }
                    // Don't reward beyond 2 wavedashes
                }
            }
            
            // 4. Reward for winning kickoff (ball goes to opponent half)
            if (pState.hasReached5050 && !kickoffResolved) {
                float ballY = state.ball.pos.y;
                float playerY = player.pos.y;
                
                // Check if ball has moved significantly from center
                if (abs(ballY) > 500.0f) {
                    kickoffResolved = true;
                    
                    // Determine which half the ball is in relative to player's side
                    bool ballOnOpponentSide = false;
                    if (player.team == Team::BLUE) {
                        // Blue team defends negative Y, so opponent side is positive Y
                        ballOnOpponentSide = (ballY > 500.0f);
                    } else {
                        // Orange team defends positive Y, so opponent side is negative Y
                        ballOnOpponentSide = (ballY < -500.0f);
                    }
                    
                    if (ballOnOpponentSide) {
                        totalReward += 2.0f; // Big reward for winning kickoff!
                    } else {
                        totalReward -= 0.5f; // Penalty for losing kickoff
                    }
                }
            }
        }
        
        // Store previous state
        pState.wasOnGround = player.isOnGround;
        
        return totalReward;
    }
    
    virtual void Reset(const GameState& initialState) override {
        playerStates.clear();
        lastBallPosY = 0.0f;
        kickoffResolved = false;
    }
};


// Simplified version focusing on dash mechanics
class KickoffDashReward : public Reward {
private:
    struct DashState {
        bool kickoffActive = false;
        bool hasFlipped = false;
        int dashCount = 0;
        bool wasOnGround = true;
        float timeInAir = 0.0f;
        Vec prevVel = Vec(0, 0, 0);
    };
    
    std::map<uint32_t, DashState> dashStates;
    const float DT = 1.0f / 120.0f;
    
    bool IsKickoff(const GameState& state) {
        float dist = sqrt(state.ball.pos.x * state.ball.pos.x + state.ball.pos.y * state.ball.pos.y);
        return dist < 50.0f && state.ball.vel.Length() < 50.0f;
    }
    
public:
    virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
        float reward = 0.0f;
        
        if (dashStates.find(player.carId) == dashStates.end()) {
            dashStates[player.carId] = DashState();
        }
        DashState& ds = dashStates[player.carId];
        
        bool isKickoff = IsKickoff(state);
        
        if (isKickoff && !ds.kickoffActive) {
            ds = DashState();
            ds.kickoffActive = true;
        }
        
        if (!isKickoff) {
            ds.kickoffActive = false;
        }
        
        if (ds.kickoffActive) {
            // Track air time
            if (!player.isOnGround) {
                ds.timeInAir += DT;
            }
            
            // Detect landing (potential wavedash)
            bool justLanded = !ds.wasOnGround && player.isOnGround;
            if (justLanded && ds.timeInAir > 0.1f && ds.timeInAir < 0.4f) {
                // Quick landing with speed = wavedash
                if (player.vel.Length() > 800.0f) {
                    if (ds.dashCount < 2) {
                        reward += (ds.dashCount == 0) ? 1.0f : 0.7f;
                        ds.dashCount++;
                    }
                }
                ds.timeInAir = 0.0f;
            }
            
            // Reset air time on ground
            if (player.isOnGround) {
                ds.timeInAir = 0.0f;
            }
            
            // Reward ball going to opponent half
            float ballY = state.ball.pos.y;
            if (abs(ballY) > 500.0f) {
                bool won = (player.team == Team::BLUE && ballY > 0) || 
                          (player.team == Team::ORANGE && ballY < 0);
                if (won) reward += 1.5f;
            }
        }
        
        ds.wasOnGround = player.isOnGround;
        ds.prevVel = player.vel;
        
        return reward;
    }
    
    virtual void Reset(const GameState& initialState) override {
        dashStates.clear();
    }
};


	class TeamSpacingReward_MKH : public Reward {
	public:
		float min_spacing;
		float penalty_strength;

		TeamSpacingReward_MKH(float min_spacing = 1000.0f, float penalty_strength = 1.0f)
			: min_spacing(min_spacing), penalty_strength(penalty_strength) {
		}

		void Reset(const GameState& initialState) override {}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float total_penalty = 0.0f;
			int count = 0;

			for (const auto& other : state.players) {
				if (other.carId == player.carId) continue;
				if (other.team != player.team) continue;

				float dist = (player.pos - other.pos).Length();

				if (dist < min_spacing) {
					float penalty = 1.0f - (dist / min_spacing);
					total_penalty += penalty;
					count++;
				}
			}

			if (count > 0) {
				return -total_penalty * penalty_strength;
			}

			return 0.0f;
		}
	};


	class DribbleReward : public Reward {
	public:
		float minBallHeight;
		float maxBallHeight;
		float maxDistance;
		float coeff;

		// Based on SwiftGroundDribbleReward - rewards speed matching between player and ball during dribbles
		DribbleReward(float minBallHeight = 109.0f, float maxBallHeight = 180.0f,
			float maxDistance = 197.0f, float coeff = 2.0f)
			: minBallHeight(minBallHeight), maxBallHeight(maxBallHeight),
			maxDistance(maxDistance), coeff(coeff) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Check all dribbling conditions
			if (!player.isOnGround) return 0.0f;
			if (state.ball.pos.z < minBallHeight || state.ball.pos.z > maxBallHeight) return 0.0f;
			if ((player.pos - state.ball.pos).Length() >= maxDistance) return 0.0f;

			// Calculate speed reward based on player-ball speed matching
			float playerSpeed = player.vel.Length();
			float ballSpeed = state.ball.vel.Length();
			float playerSpeedNormalized = playerSpeed / CommonValues::CAR_MAX_SPEED;
			float inverseDifference = 1.0f - abs(playerSpeed - ballSpeed);
			float twoSum = playerSpeed + ballSpeed;

			// Avoid division by zero
			if (twoSum == 0.0f) return 0.0f;

			float speedReward = playerSpeedNormalized + coeff * (inverseDifference / twoSum);
			return speedReward;
		}
	};

	class FlickReward : public Reward {
	public:
		float minFlickSpeed;

		// minFlickSpeed: Minimum ball speed to count as a flick
		FlickReward(float minFlickSpeed = 800.0f) : minFlickSpeed(minFlickSpeed) {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev || !player.ballTouchedStep)
				return 0.0f;

			// Check if this was a flick (ball was low, now going fast and upward)
			bool ballWasLow = state.prev->ball.pos.z < 200.0f;
			bool ballNowFast = state.ball.vel.Length() > minFlickSpeed;
			bool ballGoingUp = state.ball.vel.z > 200.0f;

			// Player should have been close to ball when touching
			float prevDistance = (player.pos - state.prev->ball.pos).Length();
			bool playerWasClose = prevDistance < 200.0f;

			if (ballWasLow && ballNowFast && ballGoingUp && playerWasClose) {
				// Reward based on ball speed after flick
				float speedRatio = RS_MIN(1.0f, state.ball.vel.Length() / 2000.0f);
				return speedRatio;
			}

			return 0.0f;
		}
	};

	class KickoffProximityReward : public Reward {
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			// Check if ball is at kickoff position (0, 0, z)
			if (abs(state.ball.pos.x) > 1.0f || abs(state.ball.pos.y) > 1.0f) {
				return 0.0f; // Not a kickoff situation
			}

			// Calculate player's distance to ball
			float playerDistToBall = (player.pos - state.ball.pos).Length();

			// Find closest opponent distance to ball
			float closestOpponentDist = FLT_MAX;
			bool foundOpponent = false;

			for (const Player& opponent : state.players) {
				// Skip teammates and self
				if (opponent.team == player.team || opponent.carId == player.carId) {
					continue;
				}

				foundOpponent = true;
				float opponentDistToBall = (opponent.pos - state.ball.pos).Length();

				if (opponentDistToBall < closestOpponentDist) {
					closestOpponentDist = opponentDistToBall;
				}
			}

			// If no opponents found, return neutral
			if (!foundOpponent) {
				return 0.0f;
			}

			// Return 1 if player is closer, -1 if opponent is closer
			return (playerDistToBall < closestOpponentDist) ? 1.0f : -1.0f;
		}
	};

	class InAirMultiTouchesReward : public Reward
	{
	private:
		// Configuration parameters
		float minPlayerHeight;
		float minBallHeight;
		float maxTimeBetweenTouches;
		float baseRewardMultiplier;
		float consecutiveMultiplier;
		float heightBonusMultiplier;
		float faceBallWeight;
		float ballToGoalWeight;
		float zVelMaintenanceWeight;
		float proximityWeight;
		int maxConsecutiveTouches;
		float maxProximityDistance;
		float targetZVel;

		// Tracking per player
		std::unordered_map<uint32_t, int> consecutiveTouches;
		std::unordered_map<uint32_t, float> lastTouchTime;
		std::unordered_map<uint32_t, bool> wasAirborne;
		std::unordered_map<uint32_t, float> airDribbleQuality;
		std::unordered_map<uint32_t, float> lastZVel;

		// Global tracking
		uint32_t lastToucherCarId;
		float currentTime;
		float lastBallZVel;

	public:
		InAirMultiTouchesReward(
			float minPlayerHeight = 256.0f,		 // Minimum height for valid air touch
			float minBallHeight = 256.0f,		 // Minimum ball height for valid air touch
			float maxTimeBetweenTouches = 1.5f,	 // Maximum time between consecutive touches
			float baseRewardMultiplier = 1.0f,	 // Base reward multiplier
			float consecutiveMultiplier = 1.5f,	 // Multiplier for consecutive touches
			float heightBonusMultiplier = 0.4f,	 // Bonus for height
			float faceBallWeight = 0.35f,		 // Weight for facing ball alignment
			float ballToGoalWeight = 0.35f,		 // Weight for ball direction to goal
			float zVelMaintenanceWeight = 0.3f,	 // Weight for Z velocity maintenance
			float proximityWeight = 0.5f,		 // Weight for proximity to ball
			int maxConsecutiveTouches = 20,		 // Maximum consecutive touches tracked
			float maxProximityDistance = 200.0f, // Maximum distance for proximity bonus
			float targetZVel = 500.0f				 // Target Z velocity to maintain (0 = floating)
		) : minPlayerHeight(minPlayerHeight),
			minBallHeight(minBallHeight),
			maxTimeBetweenTouches(maxTimeBetweenTouches),
			baseRewardMultiplier(baseRewardMultiplier),
			consecutiveMultiplier(consecutiveMultiplier),
			heightBonusMultiplier(heightBonusMultiplier),
			faceBallWeight(faceBallWeight),
			ballToGoalWeight(ballToGoalWeight),
			zVelMaintenanceWeight(zVelMaintenanceWeight),
			proximityWeight(proximityWeight),
			maxConsecutiveTouches(maxConsecutiveTouches),
			maxProximityDistance(maxProximityDistance),
			targetZVel(targetZVel),
			lastToucherCarId(0),
			currentTime(0.0f),
			lastBallZVel(0.0f)
		{
		}

		virtual void Reset(const GameState& initialState) override
		{
			consecutiveTouches.clear();
			lastTouchTime.clear();
			wasAirborne.clear();
			airDribbleQuality.clear();
			lastZVel.clear();
			lastToucherCarId = 0;
			currentTime = 0.0f;
			lastBallZVel = 0.0f;

			// Initialize tracking for all players
			for (const auto& player : initialState.players)
			{
				consecutiveTouches[player.carId] = 0;
				lastTouchTime[player.carId] = 0.0f;
				wasAirborne[player.carId] = false;
				airDribbleQuality[player.carId] = 0.0f;
				lastZVel[player.carId] = 0.0f;
			}
		}

		virtual void PreStep(const GameState& state) override
		{
			// Update current time (120 FPS)
			currentTime += 1.0f / 120.0f;

			// Track ball Z velocity
			lastBallZVel = state.ball.vel.z;

			// Check for timeouts and ground resets
			for (const auto& player : state.players)
			{
				uint32_t carId = player.carId;

				// Initialize if new player
				if (consecutiveTouches.find(carId) == consecutiveTouches.end())
				{
					consecutiveTouches[carId] = 0;
					lastTouchTime[carId] = 0.0f;
					wasAirborne[carId] = false;
					airDribbleQuality[carId] = 0.0f;
					lastZVel[carId] = 0.0f;
				}

				// Update Z velocity tracking
				lastZVel[carId] = player.vel.z;

				// Reset if too much time has passed since last touch
				if (currentTime - lastTouchTime[carId] > maxTimeBetweenTouches &&
					consecutiveTouches[carId] > 0)
				{
					ResetPlayerTouches(carId);
				}

				// Reset if player lands on ground after being airborne with touches
				if (player.isOnGround && wasAirborne[carId] && consecutiveTouches[carId] > 0)
				{
					// Give a final bonus for controlled landing with ball
					float landingBonus = 0.3f * consecutiveTouches[carId];
					airDribbleQuality[carId] += landingBonus;
					ResetPlayerTouches(carId);
				}

				// Update airborne status
				wasAirborne[carId] = !player.isOnGround;
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override
		{
			uint32_t carId = player.carId;

			// Initialize if needed
			if (consecutiveTouches.find(carId) == consecutiveTouches.end())
			{
				consecutiveTouches[carId] = 0;
				lastTouchTime[carId] = 0.0f;
				wasAirborne[carId] = false;
				airDribbleQuality[carId] = 0.0f;
				lastZVel[carId] = player.vel.z;
			}

			float reward = 0.0f;

			// Check if this is a valid air touch
			if (IsValidAirTouch(player, state))
			{
				// If this is a different player touching, reset others
				if (lastToucherCarId != 0 && lastToucherCarId != carId)
				{
					for (auto& pair : consecutiveTouches)
					{
						if (pair.first != carId && pair.second > 0)
						{
							ResetPlayerTouches(pair.first);
						}
					}
				}

				// Update consecutive touches
				consecutiveTouches[carId]++;
				consecutiveTouches[carId] = RS_MIN(consecutiveTouches[carId], maxConsecutiveTouches);
				lastTouchTime[carId] = currentTime;
				lastToucherCarId = carId;

				// Calculate air dribble quality
				float dribbleQuality = CalculateAirDribbleQuality(player, state);
				airDribbleQuality[carId] = (airDribbleQuality[carId] * 0.6f + dribbleQuality * 0.4f); // Moving average

				// Calculate reward
				reward = CalculateReward(player, state, consecutiveTouches[carId], dribbleQuality);
			}
			// Give continuous reward for maintaining good position while airborne
			else if (!player.isOnGround && player.pos.z >= minPlayerHeight &&
				consecutiveTouches[carId] > 0)
			{
				// Reward for maintaining position and alignment even without touching
				float maintenanceReward = CalculateMaintenanceReward(player, state);
				reward = maintenanceReward * 0.2f; // Smaller reward for positioning
			}

			// if (reward != 0.0f)
			// {
			// 	printf("Player %d: Z Vel Player %.2f, Ball Z Vel %.2f\n",
			// 		   carId, player.vel.z, state.ball.vel.z);
			// }

			return reward;
		}

	private:
		void ResetPlayerTouches(uint32_t carId)
		{
			consecutiveTouches[carId] = 0;
			lastTouchTime[carId] = 0.0f;
			airDribbleQuality[carId] = 0.0f;
		}

		bool IsValidAirTouch(const Player& player, const GameState& state)
		{
			// Must have touched the ball this step
			if (!player.ballTouchedStep)
			{
				return false;
			}

			// Player must be airborne
			if (player.isOnGround)
			{
				return false;
			}

			// Player must be above minimum height
			if (player.pos.z < minPlayerHeight)
			{
				return false;
			}

			// Ball must be above minimum height
			if (state.ball.pos.z < minBallHeight)
			{
				return false;
			}

			return true;
		}

		float CalculateMaintenanceReward(const Player& player, const GameState& state)
		{
			// Reward for maintaining good position without touching
			float quality = 0.0f;

			// Face ball alignment
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			float faceAlignment = RS_MAX(0.0f, player.rotMat.forward.Dot(dirToBall));
			quality += faceAlignment * 0.5f;

			// Proximity
			float distanceToBall = (player.pos - state.ball.pos).Length();
			float proximity = RS_CLAMP(1.0f - (distanceToBall / maxProximityDistance), 0.0f, 1.0f);
			quality += proximity * 0.5f;

			return quality;
		}

		float CalculateAirDribbleQuality(const Player& player, const GameState& state)
		{
			float quality = 0.0f;

			// 1. Face Ball Alignment (like FaceBallReward) - player looking at ball
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			float faceBallAlignment = RS_MAX(0.0f, player.rotMat.forward.Dot(dirToBall));
			quality += faceBallAlignment * faceBallWeight;

			// 2. Ball Direction to Goal - ball moving towards opponent's goal
			bool targetOrangeGoal = (player.team == Team::BLUE);
			Vec targetGoal = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec ballToGoalDir = (targetGoal - state.ball.pos).Normalized();
			float ballVelNorm = state.ball.vel.Length();
			float ballToGoalAlignment = 0.0f;
			if (ballVelNorm > 100.0f)
			{ // Only if ball is moving
				ballToGoalAlignment = RS_MAX(0.0f, state.ball.vel.Normalized().Dot(ballToGoalDir));
			}
			quality += ballToGoalAlignment * ballToGoalWeight;

			// 3. Z Velocity Maintenance - maintaining stable vertical velocity
			float playerZVelStability = 1.0f - RS_MIN(1.0f, std::abs(player.vel.z - targetZVel) / 500.0f);
			float ballZVelStability = 1.0f - RS_MIN(1.0f, std::abs(state.ball.vel.z - targetZVel) / 500.0f);
			float zVelSync = 1.0f - RS_MIN(1.0f, std::abs(player.vel.z - state.ball.vel.z) / 300.0f);
			float zVelQuality = (playerZVelStability * 0.3f + ballZVelStability * 0.3f + zVelSync * 0.4f);
			quality += zVelQuality * zVelMaintenanceWeight;

			return RS_CLAMP(quality, 0.0f, 1.0f);
		}

		float CalculateReward(const Player& player, const GameState& state,
			int consecutiveCount, float dribbleQuality)
		{
			if (consecutiveCount <= 0)
				return 0.0f;

			// 1. Base reward using logarithmic scale for consecutive touches
			float baseReward = baseRewardMultiplier * std::log(consecutiveCount + 1);

			// 2. Consecutive multiplier - increases with each touch
			float consecutiveBonus = std::pow(consecutiveMultiplier, RS_MIN(consecutiveCount - 1, 5));

			// 3. Height bonus - reward for maintaining altitude
			float avgHeight = (player.pos.z + state.ball.pos.z) / 2.0f;
			float heightBonus = RS_CLAMP((avgHeight - minPlayerHeight) / 1000.0f, 0.0f, 1.0f) * heightBonusMultiplier;

			// 4. Proximity bonus - critical for air dribble
			float distanceToBall = (player.pos - state.ball.pos).Length();
			float proximityBonus = RS_CLAMP(1.0f - (distanceToBall / maxProximityDistance), 0.0f, 1.0f) * proximityWeight;

			// 5. Face Ball bonus (additional to quality) - extra reward for perfect alignment
			Vec dirToBall = (state.ball.pos - player.pos).Normalized();
			float faceBallBonus = RS_MAX(0.0f, player.rotMat.forward.Dot(dirToBall));
			if (faceBallBonus > 0.9f)
			{ // Bonus for excellent alignment
				faceBallBonus *= 1.5f;
			}

			// 6. Ball to Goal bonus - extra reward if ball is going towards goal
			bool targetOrangeGoal = (player.team == Team::BLUE);
			Vec targetGoal = targetOrangeGoal ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec ballToGoalDir = (targetGoal - state.ball.pos).Normalized();
			float ballToGoalBonus = 0.0f;
			if (state.ball.vel.Length() > 100.0f)
			{
				ballToGoalBonus = RS_MAX(0.0f, state.ball.vel.Normalized().Dot(ballToGoalDir)) * 0.3f;
			}

			// 7. Z Velocity maintenance bonus - reward for stable float
			float zVelBonus = 0.0f;
			if (std::abs(player.vel.z) < 100.0f && std::abs(state.ball.vel.z) < 100.0f)
			{
				zVelBonus = 0.3f; // Bonus for maintaining float
			}

			// 8. Progressive bonus for maintaining air dribble
			float progressiveBonus = 0.0f;
			if (consecutiveCount >= 3)
			{
				progressiveBonus += 0.25f;
			}
			if (consecutiveCount >= 5)
			{
				progressiveBonus += 0.35f;
			}
			if (consecutiveCount >= 10)
			{
				progressiveBonus += 0.5f;
			}

			// 9. Quality consistency bonus - reward for maintaining high quality
			float consistencyBonus = 0.0f;
			if (airDribbleQuality[player.carId] > 0.7f && consecutiveCount >= 3)
			{
				consistencyBonus = airDribbleQuality[player.carId] * 0.6f;
			}

			// 10. Dribble quality multiplier - the most important factor
			float qualityMultiplier = 1.0f + (dribbleQuality * 2.0f); // Heavy weight on quality

			// Calculate total reward
			float totalMultiplier = consecutiveBonus * qualityMultiplier *
				(1.0f + heightBonus + proximityBonus + faceBallBonus +
					ballToGoalBonus + zVelBonus + progressiveBonus + consistencyBonus);

			float totalReward = baseReward * totalMultiplier;

			// Cap the reward to prevent exploitation
			return RS_CLAMP(totalReward, 0.0f, 20.0f);
		}
	};

	class GoodGoalPlacementReward : public Reward { //pretty good
	public:
		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) {
			Vec goalPos = (player.team == Team::BLUE) ? CommonValues::ORANGE_GOAL_CENTER : CommonValues::BLUE_GOAL_CENTER;

			float closestDist = FLT_MAX;
			Vec closestEnemyPos = goalPos;

			for (const auto& other : state.players) {
				if (other.team != player.team) {
					float dist = other.pos.Dist(goalPos);
					if (dist < closestDist) {
						closestDist = dist;
						closestEnemyPos = other.pos;
					}
				}
			}

			std::vector<float> xs = linspace(-801.505f, 801.505f, 10);
			std::vector<float> zs = linspace(91.25f, 551.525f, 10);
			float y = goalPos.y;

			float maxMinDist = -1.0f;
			Vec bestPoint = { 0, y, 0 };

			for (float x : xs) {
				for (float z : zs) {
					float minDistToAnyEnemy = FLT_MAX;
					for (const auto& other : state.players) {
						if (other.team != player.team) {
							float dist = distance2D(x, z, other.pos.x, other.pos.z);
							minDistToAnyEnemy = std::min(minDistToAnyEnemy, dist);
						}
					}
					if (minDistToAnyEnemy > maxMinDist) {
						maxMinDist = minDistToAnyEnemy;
						bestPoint = { x, y, z };
					}
				}
			}

			Vec bPos = state.ball.pos;

			float reward = 0.0f;

			if (state.goalScored && state.ball.pos.Dist(goalPos) < 2000) {
				float dstToTarget = (bPos - bestPoint).Length();
				reward = std::max(0.0f, 1.0f - (dstToTarget / 1200.0f));

				if (reward > 0.0f) {
					float velScale = std::clamp(state.ball.vel.Length() / CommonValues::BALL_MAX_SPEED, 0.0f, 1.0f);
					reward += velScale;
					reward = std::clamp(reward, 0.0f, 2.0f);
				}
			}
			return reward;
		}
	};

	class JumpTouchReward : public Reward {
	public:
		float minHeight, maxHeight, range;

		JumpTouchReward(float minHeight = 150.75f, float maxHeight = 300.0f)
			: minHeight(minHeight), maxHeight(maxHeight) {
			range = maxHeight - minHeight;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (player.ballTouchedStep && !player.isOnGround && state.ball.pos.z >= minHeight) {
				return (state.ball.pos.z - minHeight) / range;
			}
			return 0;
		}
	};

	inline Vec FindBestGoalPoint(const Player& player, const GameState& state)
	{
		const Vec goalPos = (player.team == Team::BLUE)
			? CommonValues::ORANGE_GOAL_CENTER
			: CommonValues::BLUE_GOAL_CENTER;

		auto isOpponent = [&](const Player& me, const Player& other) {
			return me.team != other.team;
			};

		// search grid inside the goal
		const std::vector<float> xs = linspace(-801.505f, 801.505f, 10);
		const std::vector<float> zs = linspace(91.25f, 551.525f, 10);
		const float y = goalPos.y;

		float bestMinDst = -1.f;
		Vec   bestPoint = { 0, y, 0 };

		for (float x : xs)
			for (float z : zs) {
				float minToEnemy = FLT_MAX;
				for (const auto& other : state.players)
					if (isOpponent(player, other))
						minToEnemy = std::min(minToEnemy,
							distance2D(x, z, other.pos.x, other.pos.z));

				if (minToEnemy > bestMinDst) {
					bestMinDst = minToEnemy;
					bestPoint = { x, y, z };
				}
			}
		return bestPoint;
	}

	class AirdribbleRewardV1 : public Reward
	{
	public:
		AirdribbleRewardV1()
			: lastPlayerZ(0.0f), lastBallZ(0.0f)
		{
		}

		virtual void Reset(const GameState& initialState) override
		{
			lastPlayerZ = 0.0f;
			lastBallZ = 0.0f;
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override
		{
			float reward = 0.0f;
			constexpr float SIDE_WALL_X = 4096.0f;

			// --- Distance to ball ---
			Vec posDiff = state.ball.pos - player.pos;
			float distToBall = posDiff.Length();

			// --- Facing ball ---
			Vec normPosDiff = (posDiff.Length() > 0) ? (posDiff / posDiff.Length()) : posDiff;
			float facingBall = player.rotMat.forward.Dot(normPosDiff);

			// --- Ball position geometry ---
			float BallY = RS_MAX(state.ball.pos.y, 0.0f);
			float NewY = 6000.0f - BallY;
			float BallX = std::abs(state.ball.pos.x) + 92.75f;
			float LargestX = NewY * 0.683f;

			// --- Air roll reward (bias term) ---
			float Airroll = (player.angVel.y / 5.5f) * 5.0f;

			// --- Align ball to goal ---
			Vec protecc = CommonValues::BLUE_GOAL_BACK;
			Vec attacc = CommonValues::ORANGE_GOAL_BACK;
			if (player.team == Team::ORANGE)
				std::swap(protecc, attacc);

			Vec toBall = state.ball.pos - player.pos;
			Vec toGoal = attacc - player.pos;
			float lenA = toBall.Length(), lenB = toGoal.Length();
			float offensiveReward = (lenA > 0 && lenB > 0) ? (toBall.Dot(toGoal) / (lenA * lenB)) : 0.f;

			// --- Air dribble detection ---
			if (!player.isOnGround &&
				state.ball.pos.z > 250.0f &&
				player.pos.z < state.ball.pos.z &&
				state.ball.pos.y > 1000.0f &&
				BallX < LargestX)
			{
				bool ascending = (player.pos.z > lastPlayerZ && state.ball.pos.z > lastBallZ);

				// CASE 1: ascending
				if (ascending)
				{
					if (distToBall < 400.0f && player.boost > 0.30f && facingBall > 0.74f)
					{
						if (player.pos.y < state.ball.pos.y)
						{
							if (player.ballTouchedStep)
								reward += 40.0f;
							reward += 5.0f;
							reward += Airroll * 2.5f;
						}
						reward += facingBall;
						reward += (1.0f - (distToBall / 400.0f)) * 7.5f;
					}
					else if (distToBall < 400.0f && facingBall > 0.74f)
					{
						if (player.pos.y < state.ball.pos.y)
						{
							if (player.ballTouchedStep)
								reward += 20.0f;
							reward += 2.5f;
							if (player.boost > 0.0f)
								reward += Airroll * 1.25f;
						}
						reward += facingBall / 2.0f;
						reward += (1.0f - (distToBall / 400.0f)) * (7.5f / 2.0f);
					}
				}
				// CASE 2: not ascending
				else
				{
					if (distToBall < 400.0f && player.boost > 0.15f && facingBall > 0.74f)
					{
						if (player.pos.y < state.ball.pos.y)
						{
							if (player.ballTouchedStep)
								reward += 40.0f;
							reward += 5.0f;
							reward += Airroll * 2.5f;
						}
						reward += facingBall;
						reward += (1.0f - (distToBall / 400.0f)) * 7.5f;
					}
					else if (distToBall < 400.0f && facingBall > 0.74f)
					{
						if (player.pos.y < state.ball.pos.y)
						{
							if (player.ballTouchedStep)
								reward += 20.0f;
							reward += 2.5f;
							if (player.boost > 0.0f)
								reward += Airroll * 1.25f;
						}
						reward += facingBall / 2.0f;
						reward += (1.0f - (distToBall / 400.0f)) * (7.5f / 2.0f);
					}
					reward /= 2.0f;
				}
			}

			// --- Store last frame Z-values ---
			lastPlayerZ = player.pos.z;
			lastBallZ = state.ball.pos.z;

			// --- Ball height scaling ---
			float ballZ = state.ball.pos.z;
			constexpr float GOAL_HEIGHT = 642.775f;
			constexpr float BALL_RADIUS = 92.75f;

			if (ballZ > 2000.0f)
				ballZ = 2000.0f;
			else if (ballZ < (GOAL_HEIGHT - (BALL_RADIUS * 3.5f)))
				ballZ = 0.0f;

			// --- Combine offensive alignment & height bonus ---
			reward *= offensiveReward;

			if (!player.isOnGround && distToBall < 400.0f && facingBall > 0.7f)
				reward += (ballZ / 50.0f);


			// --- Normalize final reward ---
			return reward / 100.0f;
		}

	private:
		float lastPlayerZ;
		float lastBallZ;
	};

	class AirdribbleRewardV2 : public Reward
	{
	public:
		const float CROSSBAR_HEIGHT = 642.775f; // The height of the goal
		const float BALL_RADIUS = 91.25f;

	public:
		AirdribbleRewardV2() {}

		virtual void Reset(const GameState& initialState) override {}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override
		{
			Vec protecc = CommonValues::BLUE_GOAL_BACK;
			Vec attacc = CommonValues::ORANGE_GOAL_BACK;
			if (player.team == Team::ORANGE)
				std::swap(protecc, attacc);

			// --- 1. Activation Check ---
			float distToBall = (state.ball.pos - player.pos).Length();
			// Must be in air, close to ball, and ball must be ABOVE crossbar
			if (player.isOnGround || distToBall > 600.f || state.ball.pos.z < CROSSBAR_HEIGHT) {
				return 0.0f; // Ball is too low = Easy save = 0 points
			}

			// --- 2. PROGRESS REWARD (SPEED) ---
			// Since we know the ball is high enough, we just reward SPEED.
			Vec ballToGoalDir = (attacc - state.ball.pos).Normalized();
			float ballSpeedTowardsGoal = ballToGoalDir.Dot(state.ball.vel);
			float progressReward = RS_MAX(0.f, ballSpeedTowardsGoal / CommonValues::BALL_MAX_SPEED);

			// --- 3. MAINTENANCE REWARD ---
			Vec normPosDiff = (state.ball.pos - player.pos).Normalized();
			float facingBall = player.rotMat.forward.Dot(normPosDiff);
			float proximityReward = (1.0f - (distToBall / 600.0f));

			float maintenanceReward = RS_MAX(0.f, facingBall) * proximityReward;

			if (player.ballTouchedStep) {
				maintenanceReward += 0.5f;
			}

			// Combine: Speed is main driver
			// (1.0 * 5.0) + (1.5 * 1.0) = ~6.5
			float finalReward = (progressReward * 5.0f) + (maintenanceReward * 1.0f);

			return finalReward;
		}
	};

	class AirRollRewardV2 : public Reward {
	private:
		struct AirRollState {
			float lastDir = 0.0f;
			float switchPenalty = 0.0f;
			uint64_t lastTickSeen = 0ULL;
			uint64_t lastRollTick = 0ULL;
		};

	public:
		AirRollRewardV2(
			float rewardRight = 1.5f,
			float rewardLeft = 0.5f,
			float minHeight = 320.0f,
			float maxBallDistance = 1000.0f,
			float alignmentThreshold = 0.2f,
			float rollThreshold = 0.5f,
			float minApproachSpeed = 300.0f,
			float fastApproachMultiplier = 1.01f,
			float fastApproachAlignment = 0.01f,
			float switchPenaltyIncrement = 1.0f,
			float switchPenaltyDecayPerTick = 0.05f,
			float penaltyScale = 0.5f,
			uint64_t directionResetTicks = 60ULL
		) :
			_rewardRight(rewardRight),
			_rewardLeft(rewardLeft),
			_minHeight(minHeight),
			_maxBallDistance(std::max(0.0f, maxBallDistance)),
			_alignmentThreshold(alignmentThreshold),
			_rollThreshold(rollThreshold),
			_minApproachSpeed(minApproachSpeed),
			_fastApproachMultiplier(std::max(1.0f, fastApproachMultiplier)),
			_fastApproachAlignment(std::max(0.0f, fastApproachAlignment)),
			_switchPenaltyIncrement(std::max(0.0f, switchPenaltyIncrement)),
			_switchPenaltyDecayPerTick(std::max(0.0f, switchPenaltyDecayPerTick)),
			_penaltyScale(std::max(0.0f, penaltyScale)),
			_directionResetTicks(directionResetTicks) {
			if (_directionResetTicks == 0ULL) {
				_directionResetTicks = 60ULL;
			}
		}

		void Reset(const GameState& initialState) override {
			_states.clear();
		}

		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			AirRollState& st = _states[player.carId];
			_decayState(st, state.lastTickCount);

			if (player.isOnGround)
				return 0.0f;

			if (player.pos.z < _minHeight || state.ball.pos.z < _minHeight)
				return 0.0f;

			Vec toBall = state.ball.pos - player.pos;
			float distToBall = toBall.Length();
			if (distToBall < 1e-3f)
				return 0.0f;
			if (_maxBallDistance > 0.0f && distToBall > _maxBallDistance)
				return 0.0f;
			Vec dirToBall = toBall / distToBall;

			float speed = player.vel.Length();
			if (speed < 1e-3f)
				return 0.0f;

			float approachSpeed = dirToBall.Dot(player.vel);
			if (approachSpeed < _minApproachSpeed)
				return 0.0f;

			float alignment = approachSpeed / (speed + 1e-6f);
			if (alignment < _alignmentThreshold) {
				bool fastApproach = approachSpeed >= _minApproachSpeed * _fastApproachMultiplier;
				if (!(fastApproach && alignment >= _fastApproachAlignment))
					return 0.0f;
			}

			float rollInput = player.prevAction[4];
			if (std::fabs(rollInput) < _rollThreshold)
				return 0.0f;

			float direction = rollInput > 0.0f ? 1.0f : -1.0f;
			if (st.lastDir != 0.0f && direction != st.lastDir) {
				st.switchPenalty = std::min(6.0f, st.switchPenalty + _switchPenaltyIncrement);
			}
			st.lastDir = direction;
			st.lastRollTick = state.lastTickCount;

			float penaltyMultiplier = 1.0f;
			if (_penaltyScale > 0.0f) {
				penaltyMultiplier = std::max(0.0f, 1.0f - _penaltyScale * st.switchPenalty);
			}

			float baseReward = direction > 0.0f ? _rewardRight : _rewardLeft;
			return baseReward * penaltyMultiplier;
		}

	private:
		void _decayState(AirRollState& st, uint64_t currentTick) {
			if (st.lastTickSeen == 0ULL) {
				st.lastTickSeen = currentTick;
			}
			else if (currentTick > st.lastTickSeen) {
				uint64_t deltaTicks = currentTick - st.lastTickSeen;
				if (_switchPenaltyDecayPerTick > 0.0f) {
					float decayAmount = _switchPenaltyDecayPerTick * static_cast<float>(deltaTicks);
					st.switchPenalty = std::max(0.0f, st.switchPenalty - decayAmount);
				}
				st.lastTickSeen = currentTick;
			}

			if (_directionResetTicks > 0ULL && st.lastRollTick > 0ULL && currentTick > st.lastRollTick) {
				uint64_t ticksSinceRoll = currentTick - st.lastRollTick;
				if (ticksSinceRoll >= _directionResetTicks && st.switchPenalty <= 1e-3f) {
					st.lastDir = 0.0f;
					st.lastRollTick = 0ULL;
				}
			}
		}

		float _rewardRight;
		float _rewardLeft;
		float _minHeight;
		float _maxBallDistance;
		float _alignmentThreshold;
		float _rollThreshold;
		float _minApproachSpeed;
		float _fastApproachMultiplier;
		float _fastApproachAlignment;
		float _switchPenaltyIncrement;
		float _switchPenaltyDecayPerTick;
		float _penaltyScale;
		uint64_t _directionResetTicks;
		std::unordered_map<uint32_t, AirRollState> _states;
	};

	class DribbleBumpReward : public Reward {
	public:
		float dribbleDistance;
		float carBallHeightDiff;
		float maxTimeSinceDribble;
		float baseReward;
		float speedBonus;

		std::unordered_map<uint32_t, bool> isDribbling;
		std::unordered_map<uint32_t, float> timeSinceDribble;
		std::unordered_map<uint32_t, float> lastDribbleSpeed;

		DribbleBumpReward(
			float dribbleDistance = 197.0f,
			float carBallHeightDiff = 110.0f,
			float maxTimeSinceDribble = 2.5f,
			float baseReward = 1.0f,
			float speedBonus = 0.5f
		) : dribbleDistance(dribbleDistance),
			carBallHeightDiff(carBallHeightDiff),
			maxTimeSinceDribble(maxTimeSinceDribble),
			baseReward(baseReward),
			speedBonus(speedBonus) {
		}

		virtual void Reset(const GameState& initialState) override {
			isDribbling.clear();
			timeSinceDribble.clear();
			lastDribbleSpeed.clear();

			for (const auto& player : initialState.players) {
				isDribbling[player.carId] = false;
				timeSinceDribble[player.carId] = 999.0f;
				lastDribbleSpeed[player.carId] = 0.0f;
			}
		}

		virtual void PreStep(const GameState& state) override {
			for (const auto& player : state.players) {
				uint32_t carId = player.carId;

				Vec ballCarVec = state.ball.pos - player.pos;
				bool isCurrentlyDribbling = (
					ballCarVec.Length() < dribbleDistance &&
					state.ball.pos.z > (player.pos.z + carBallHeightDiff)
					);

				if (isDribbling[carId] && !isCurrentlyDribbling) {
					timeSinceDribble[carId] = 0.0f;
					lastDribbleSpeed[carId] = player.vel.Length();
				}
				else if (isCurrentlyDribbling) {
					timeSinceDribble[carId] = 999.0f;
				}
				else if (timeSinceDribble[carId] < 999.0f) {
					timeSinceDribble[carId] += state.deltaTime;
				}

				isDribbling[carId] = isCurrentlyDribbling;
			}
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			uint32_t carId = player.carId;

			if (!player.eventState.bump) {
				return 0.0f;
			}

			if (timeSinceDribble[carId] > maxTimeSinceDribble) {
				return 0.0f;
			}

			float reward = baseReward;

			float timeFactor = 1.0f - (timeSinceDribble[carId] / maxTimeSinceDribble);
			reward *= (1.0f + timeFactor * 0.5f);

			float speedFactor = lastDribbleSpeed[carId] / CommonValues::CAR_MAX_SPEED;
			reward *= (1.0f + speedFactor * speedBonus);

			float ballDistance = (state.ball.pos - player.pos).Length();
			if (ballDistance < dribbleDistance * 1.5f) {
				reward *= 1.25f;
			}

			if (player.eventState.demo) {
				reward *= 1.5f;
			}

			return RS_CLAMP(reward, 0.0f, 3.0f);
		}
	};

	class RefinedDemoReward : public Reward {
	private:
		float max_demo_angle;
		float optimal_demo_angle;
		float base_reward;
		float flip_bonus;
		float max_reward;
		int flip_window_ticks;

		struct OpponentState { bool was_demoed = false; };
		struct PlayerFlipState { int last_flip_tick = -999; bool prev_has_flip = true; bool prev_jump_pressed = false; };

		std::unordered_map<uint32_t, OpponentState> opponent_states;
		std::unordered_map<uint32_t, PlayerFlipState> player_flip_states;
		int current_tick;

	public:
		// Constructor Implementation
		RefinedDemoReward(
			float max_demo_angle_deg = 70.0f,
			float optimal_demo_angle_deg = 45.0f,
			float base_reward_ = 1.0f,
			float flip_bonus_ = 1.5f,
			float max_reward_ = 3.0f,
			int flip_window_ticks_ = 12)
			: max_demo_angle(max_demo_angle_deg* (M_PI / 180.0f)),
			optimal_demo_angle(optimal_demo_angle_deg* (M_PI / 180.0f)),
			base_reward(base_reward_),
			flip_bonus(flip_bonus_),
			max_reward(max_reward_),
			flip_window_ticks(flip_window_ticks_),
			current_tick(0) {
		}

		// Reset Implementation
		void Reset(const GameState& initialState) override {
			opponent_states.clear();
			player_flip_states.clear();
			current_tick = 0;

			for (const auto& p : initialState.players) {
				opponent_states[p.carId].was_demoed = p.isDemoed;
				player_flip_states[p.carId] = PlayerFlipState();
			}
		}

		// GetReward Implementation with LESS SPAMMY Logging (Only prints on Demo/Missed Demo)
		float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float reward = 0.0f;
			uint32_t carId = player.carId;
			current_tick++;

			auto& flipState = player_flip_states[carId];
			const Action& action = player.prevAction;
			bool jump_pressed = (action.jump > 0.5f);
			bool has_flip = !player.hasFlipped && !player.isOnGround && player.hasJumped;

			// Flip detection is still needed for reward calculation, but we won't log it every time.
			if (jump_pressed && !flipState.prev_jump_pressed && has_flip) {
				flipState.last_flip_tick = current_tick;
			}

			flipState.prev_jump_pressed = jump_pressed;
			flipState.prev_has_flip = has_flip;

			Vec car_pos = player.pos;
			Vec car_forward = player.rotMat.forward;
			float car_speed = player.vel.Length();
			bool is_supersonic = (car_speed >= 2200.0f);

			// Check all opponents for demos
			for (const auto& opponent : state.players) {
				if (opponent.team == player.team) continue;

				uint32_t opp_id = opponent.carId;
				auto& opp_state = opponent_states[opp_id];
				bool was_demoed = opp_state.was_demoed;
				bool now_demoed = opponent.isDemoed;

				// *** CORE LOGIC & LOGGING TRIGGER: Only triggers when a DEMO state change happens. ***
				if (!was_demoed && now_demoed) {
					Vec to_enemy = opponent.pos - car_pos;
					Vec to_enemy_flat(to_enemy.x, to_enemy.y, 0.0f);
					float to_enemy_dist = to_enemy_flat.Length();

					if (to_enemy_dist > 1e-6f) {
						Vec to_enemy_dir = to_enemy_flat / to_enemy_dist;
						Vec car_fwd_flat(car_forward.x, car_forward.y, 0.0f);
						float car_fwd_dist = car_fwd_flat.Length();
						if (car_fwd_dist < 1e-6f) car_fwd_dist = 1e-6f;
						Vec car_fwd_dir = car_fwd_flat / car_fwd_dist;

						float dot_product = car_fwd_dir.Dot(to_enemy_dir);
						dot_product = std::max(-1.0f, std::min(1.0f, dot_product));
						float angle = std::acos(dot_product);
						float angle_deg = angle * 180.0f / M_PI;

						if (angle <= max_demo_angle) {
							float angle_score = std::max(0.0f, 1.0f - (angle / max_demo_angle));
							int ticks_since_flip = current_tick - flipState.last_flip_tick;
							bool flipped_recently = (ticks_since_flip <= flip_window_ticks);

							if (flipped_recently && !is_supersonic) {
								reward = std::min(max_reward, base_reward + flip_bonus + angle_score);
								// LOGGING: HIGH-VALUE EVENT
								std::cout << "[DemoReward] 🚀 FLIP DEMO: P" << carId << " demoed O" << opp_id
									<< " | Angle: " << angle_deg << "° | Reward: " << reward << std::endl;
							}
							else {
								reward = base_reward + angle_score;
								// LOGGING: HIGH-VALUE EVENT
								std::cout << "[DemoReward] 💥 REGULAR DEMO: P" << carId << " demoed O" << opp_id
									<< " | Supersonic: " << (is_supersonic ? "Y" : "N")
									<< " | Reward: " << reward << std::endl;
							}
						}
						else {
							// LOGGING: Important failure case
							std::cout << "[DemoReward] ❌ MISSED REWARD: P" << carId << " demoed O" << opp_id
								<< " but angle was " << angle_deg << "° (Max: " << (max_demo_angle * 180.0f / M_PI) << "°)." << std::endl;
						}
					}
				}

				// Update opponent demo status for next tick
				opp_state.was_demoed = now_demoed;
			}

			return reward;
		}
	};

	class PressureFlickReward : public Reward {
	public:
		const float PANIC_DISTANCE = 700.0f;
		const float MIN_FLICK_SPEED = 1000.0f;
		const float TARGET_FLICK_SPEED = 2920.0f;
		const float EXPONENT = 2.5f;

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!player.ballTouchedStep || !player.isFlipping) return 0.0f;

			if (!state.prev) return 0.0f;

			// Distance
			float dist = player.prev->pos.Dist(state.prev->ball.pos);
			if (dist > 250.0f) return 0.0f;

			// Speed Match
			float speedDiff = std::abs(player.prev->vel.Length() - state.prev->ball.vel.Length());
			if (speedDiff > 500.0f) return 0.0f;

			// Distance Check
			float closestOppDist = 100000.0f;
			for (const auto& p : state.players) {
				if (p.team != player.team) {
					float d = player.pos.Dist(p.pos);
					if (d < closestOppDist) closestOppDist = d;
				}
			}
			if (closestOppDist > PANIC_DISTANCE) return 0.0f;

			// Result Check
			bool targetOrange = player.team == Team::BLUE;
			Vec targetPos = targetOrange ? CommonValues::ORANGE_GOAL_BACK : CommonValues::BLUE_GOAL_BACK;
			Vec dirToGoal = (targetPos - state.ball.pos).Normalized();
			float velTowardsGoal = state.ball.vel.Dot(dirToGoal);

			if (velTowardsGoal > MIN_FLICK_SPEED) {
				float ratio = velTowardsGoal / TARGET_FLICK_SPEED;
				float reward = std::pow(ratio, EXPONENT);
				return std::min(reward, 2.0f);
			}

			return 0.0f;
		}
	};

	class FlyToGoal : public Reward {
	public:
		// ============================================================================
		// PARAMETRI BASE
		// ============================================================================


		// --- Gating & reward base (per-tick) ---
		float min_air_height = 320.0f;
		float max_ball_dist = 450.0f;              // Leggermente aumentato (era 380)
		float per_tick_scale = 0.08f;              // Aumentato da 0.05

		// --- Touch bonus (esponenziale verso porta) ---
		float touch_base = 10.0f;                  // Aumentato da 15
		float lambda_goal = 5.0f;                 // Più aggressivo (era 10)
		float min_touch_factor = 0.0f;             // ZERO per tocchi sbagliati (era 0.02)
		float max_delta_v = 2500.0f;
		bool  use_impulse_delta = true;

		// --- Air roll bonus (BILANCIATO) ---
		float airroll_bonus_base = 0.4f;          // Ridotto da 0.2
		float airroll_bonus_max = 2.5f;           //  RIDOTTO da 2.3 (cruciale!)
		float min_airroll_rate = 0.04f;          // Soglia più alta (era 0.08)
		float airroll_speed_threshold = 200.0f;    // Leggermente più alto

		int   max_progression_ticks = 40;          // Più lento (era 40)
		int   min_ticks_for_bonus = 10;          // Più stringente (era 8)

		int   decay_rate_active = 4;              // Più veloce (era 3)
		int   decay_rate_slow = 3;              // Più veloce (era 2)
		int   decay_rate_gating = 12;             // Più aggressivo (era 12)

		float smooth_factor_active = 0.70f;
		float smooth_factor_inactive = 0.40f;

		float quality_threshold = 0.8f;      // Più stringente (era 0.70)
		float quality_multiplier = 2.2f;       //  RIDOTTO da 2.5
		float consistency_bonus_rate = 0.7f;       //  RIDOTTO da 0.7
		int   consistency_start_ticks = 15;        // Più tardi (era 20)

		float direction_detect_mult = 2.0f;

		// --- Field position scaling ---
		float min_field_mult = 0.35f;              // Più permissivo (era 0.27)
		float max_field_mult = 1.35f;              // Leggermente più alto
		float optimal_start_dist = 6000.0f;
		float too_far_penalty_dist = 10000.0f;
		bool  use_field_scaling = true;

		// --- Bonus "sotto la palla" (solo per touch) ---
		float under_ball_min_gap = 180.0f;         // Leggermente aumentato
		float under_ball_max_gap = 450.0f;
		float under_touch_boost = 0.5f;          //  RIDOTTO da 0.6

		// ============================================================================
		// PARAMETRI ALLINEAMENTO DIREZIONE TARGET (dal documento)
		// ============================================================================
		float w_dir_ball = 0.35f;                // Peso verso la palla
		float w_dir_goal = 0.75f;                // Peso verso la porta (palla→porta)
		float min_goal_push_dot = 0.30f;           //  AUMENTATO da 0.15 (più stringente!)
		float cross_track_penalty = 0.20f;         //  AUMENTATO da 0.06 (3x più forte)
		float opposite_dir_penalty = 0.35f;        //  AUMENTATO da 0.12 (3x più forte)

		// ============================================================================
		// 🆕 PARAMETRI SHOT QUALITY (nuovo sistema semplificato)
		// ============================================================================
		float shot_quality_bonus = 8.0f;           // Bonus per posizionamento ottimale
		float optimal_approach_angle = 25.0f;      // Angolo ottimale approach (gradi)
		float shot_quality_distance_max = 0.8f;    // Max dist come % di max_ball_dist

		// 🆕 Height optimization
		float optimal_shot_height = 380.0f;        // Altezza ottimale per shot
		float shot_height_bonus = 4.0f;            // Bonus aggiuntivo per altezza corretta

		// ============================================================================
		// STATE TRACKING
		// ============================================================================
		struct PlayerAirRollState {
			int   consecutive_ticks = 0;
			float prev_roll_direction = 0.0f;
			float smoothed_roll_rate = 0.0f;
			float accumulated_bonus = 0.0f;
			int   stable_direction_ticks = 0;
			float total_airroll_time = 0.0f;
			bool  has_reached_min_ticks = false;

			// 🆕 Shot quality tracking
			float best_shot_quality = 0.0f;
			int   ticks_in_good_position = 0;
		};
		std::unordered_map<uint32_t, PlayerAirRollState> player_states;

		// ============================================================================
		// HELPER FUNCTIONS
		// ============================================================================

		inline static float clamp01(float x) {
			return std::max(0.0f, std::min(1.0f, x));
		}

		std::string GetName() override {
			return "FlyToGoal";
		}

		void Reset(const GameState& initialState) override {
			player_states.clear();
		}

		float GetFieldPositionMultiplier(const Player& player, const Vec& ball_pos) {
			if (!use_field_scaling) return 1.0f;

			Vec oppGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;
			Vec ownGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::BLUE_GOAL_CENTER
				: CommonValues::ORANGE_GOAL_CENTER;

			float dist_to_opp_goal = (ball_pos - oppGoal).Length();
			float field_mult = 1.0f;

			if (dist_to_opp_goal <= optimal_start_dist) {
				float proximity_factor = 1.0f - (dist_to_opp_goal / optimal_start_dist);
				field_mult = 1.0f + (max_field_mult - 1.0f) * proximity_factor;
			}
			else if (dist_to_opp_goal <= too_far_penalty_dist) {
				float penalty_factor = (dist_to_opp_goal - optimal_start_dist) /
					(too_far_penalty_dist - optimal_start_dist);
				field_mult = 1.0f - (1.0f - min_field_mult) * penalty_factor;
			}
			else {
				field_mult = min_field_mult;
			}

			// Penalità own half ridotta
			float midfield_y = (oppGoal.y + ownGoal.y) / 2.0f;
			bool in_own_half = (player.team == RocketSim::Team::BLUE && ball_pos.y < midfield_y) ||
				(player.team == RocketSim::Team::ORANGE && ball_pos.y > midfield_y);
			if (in_own_half) {
				field_mult *= 0.85f;  // Era 0.7, ora più permissivo
			}

			return std::max(min_field_mult, std::min(max_field_mult, field_mult));
		}

		// 🆕 NUOVO: Calcola qualità posizionamento per shot
		float GetShotQuality(const Vec& player_pos, const Vec& ball_pos,
			const Vec& ball_to_goal_dir, float ball_height) {
			// Approach angle quality
			Vec player_to_ball = ball_pos - player_pos;
			player_to_ball.z = 0.0f;  // Solo orizzontale
			float dist = player_to_ball.Length();

			if (dist < 1e-6f) return 0.5f;

			Vec approach_dir = player_to_ball / dist;
			float dot = approach_dir.Dot(ball_to_goal_dir);
			dot = std::max(-1.0f, std::min(1.0f, dot));

			float angle_rad = std::acos(dot);
			float angle_deg = angle_rad * 180.0f / 3.14159265f;

			// Premia approach laterale/dietro palla (migliore per shot)
			float angle_quality = 0.0f;
			if (angle_deg < optimal_approach_angle) {
				angle_quality = 1.0f;
			}
			else if (angle_deg < optimal_approach_angle + 40.0f) {
				float ratio = (angle_deg - optimal_approach_angle) / 40.0f;
				angle_quality = 1.0f - 0.6f * ratio;
			}
			else {
				angle_quality = 0.4f;
			}

			// Height quality (premia altezze ottimali per shot)
			float height_diff = std::abs(ball_height - optimal_shot_height);
			float height_quality = 1.0f;
			if (height_diff > 400.0f) {
				height_quality = 0.5f;
			}
			else {
				height_quality = 1.0f - 0.5f * (height_diff / 400.0f);
			}

			return angle_quality * height_quality;
		}

		// ============================================================================
		// MAIN REWARD FUNCTION
		// ============================================================================

		float GetReward(const Player& player, const GameState& state, bool /*isFinal*/) override {
			auto& ar_state = player_states[player.carId];

			// ========================================================================
			// GATING
			// ========================================================================
			bool passed_gates = true;
			if (player.isOnGround || player.pos.z < min_air_height) {
				passed_gates = false;
			}

			float dist = (state.ball.pos - player.pos).Length();
			if (dist > max_ball_dist) {
				passed_gates = false;
			}

			if (!passed_gates) {
				// Decay aggressivo quando fuori dal gating
				ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - decay_rate_gating);
				ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_gating);
				ar_state.smoothed_roll_rate *= 0.4f;
				ar_state.has_reached_min_ticks = false;
				ar_state.best_shot_quality = 0.0f;
				ar_state.ticks_in_good_position = 0;
				return 0.0f;
			}

			float field_mult = GetFieldPositionMultiplier(player, state.ball.pos);

			// ========================================================================
			// DIREZIONI BASE
			// ========================================================================
			Vec toBall = state.ball.pos - player.pos;
			Vec toBall2D(toBall.x, toBall.y, 0.0f);
			float d2 = toBall2D.Length();
			Vec dirToBall2D = (d2 > 1e-6f) ? (toBall2D / d2) : Vec(0, 0, 0);

			Vec v2D(player.vel.x, player.vel.y, 0.0f);
			float v2 = v2D.Length();
			Vec vHat2D = (v2 > 1e-6f) ? (v2D / v2) : Vec(0, 0, 0);

			// ========================================================================
			// DIREZIONE TARGET BLEND (palla→porta + verso palla)
			// ========================================================================
			Vec oppGoal = (player.team == RocketSim::Team::BLUE)
				? CommonValues::ORANGE_GOAL_CENTER
				: CommonValues::BLUE_GOAL_CENTER;

			Vec ballToGoal2D(oppGoal.x - state.ball.pos.x, oppGoal.y - state.ball.pos.y, 0.0f);
			float bgL = ballToGoal2D.Length();
			Vec dirBallToGoal2D = (bgL > 1e-6f) ? (ballToGoal2D / bgL) : Vec(0, 0, 0);

			// Blend dinamico: più vicino alla palla → più peso verso palla
			float nearBall01 = clamp01((max_ball_dist - std::min(max_ball_dist, d2)) / max_ball_dist);
			float wB = w_dir_ball * (0.6f + 0.4f * nearBall01);
			float wG = w_dir_goal * (1.0f - 0.3f * nearBall01);

			Vec desired2D = wB * dirToBall2D + wG * dirBallToGoal2D;
			float desL = desired2D.Length();
			if (desL > 1e-6f) desired2D = desired2D / desL;

			// Allineamento velocità e muso alla direzione target
			Vec fwd2D(player.rotMat.forward.x, player.rotMat.forward.y, 0.0f);
			float f2 = fwd2D.Length();
			if (f2 > 1e-6f) fwd2D = fwd2D / f2;

			float align_vel_desired = clamp01(vHat2D.Dot(desired2D));
			float align_fwd_desired = clamp01(fwd2D.Dot(desired2D));

			// Cross-track: componente laterale rispetto a desired2D
			float cross = std::sqrt(std::max(0.0f, 1.0f - align_vel_desired * align_vel_desired));
			float cross_pen = cross_track_penalty * cross;

			// Penalità se vai contro la direzione target
			float opposite_pen = 0.0f;
			if (v2 > 50.0f) {
				float rawDot = vHat2D.Dot(desired2D);
				if (rawDot < 0.0f) {
					opposite_pen = opposite_dir_penalty * (-rawDot);
				}
			}

			// ========================================================================
			// ALLINEAMENTI "CLASSICI" (per compatibilità)
			// ========================================================================
			float vlen = player.vel.Length();
			Vec vHat = (vlen > 1e-6f) ? (player.vel / vlen) : Vec(0, 0, 0);
			Vec dirToBall = (dist > 1e-6f) ? ((state.ball.pos - player.pos) / dist) : Vec(0, 0, 0);
			float align_ball = clamp01(vHat.Dot(dirToBall));

			const float R = CommonValues::BALL_RADIUS;
			float proximity = clamp01(1.0f - (dist - R) / std::max(1.0f, (max_ball_dist - R)));

			// ========================================================================
			// REWARD BASE PER-TICK (weighted blend)
			// ========================================================================
			float shaped_core =
				0.50f * align_vel_desired +   // Direzione target
				0.20f * align_fwd_desired +   // Muso allineato
				0.20f * align_ball +          // Ancora verso palla
				0.10f * proximity;            // Prossimità

			shaped_core = clamp01(shaped_core);
			float reward = per_tick_scale * shaped_core * field_mult;

			// Applica penalità (aumentate!)
			reward -= (cross_pen + opposite_pen) * field_mult;

			// ========================================================================
			// 🆕 SHOT QUALITY BONUS
			// ========================================================================
			if (dist < max_ball_dist * shot_quality_distance_max) {
				float shot_quality = GetShotQuality(player.pos, state.ball.pos,
					dirBallToGoal2D, state.ball.pos.z);

				// Reward continuo per posizione ottimale
				reward += shot_quality_bonus * shot_quality * field_mult * 0.04f;

				// Track best position
				ar_state.best_shot_quality = std::max(ar_state.best_shot_quality, shot_quality);

				if (shot_quality > 0.75f) {
					ar_state.ticks_in_good_position++;

					// Bonus milestone per mantenere posizione ottimale
					if (ar_state.ticks_in_good_position == 15) {
						reward += 2.0f * field_mult;
					}
					else if (ar_state.ticks_in_good_position == 30) {
						reward += 3.0f * field_mult;
					}
				}
				else {
					ar_state.ticks_in_good_position = 0;
				}

				// Bonus extra per altezza ottimale
				float height_diff = std::abs(state.ball.pos.z - optimal_shot_height);
				if (height_diff < 150.0f) {
					float height_quality = 1.0f - (height_diff / 150.0f);
					reward += shot_height_bonus * height_quality * field_mult * 0.03f;
				}
			}

			// ========================================================================
			// TOUCH BONUS (esponenziale verso porta, ZERO per tocchi sbagliati)
			// ========================================================================
			if (player.ballTouchedStep) {
				Vec dV = (use_impulse_delta && state.prev)
					? (state.ball.vel - state.prev->ball.vel)
					: state.ball.vel;

				Vec dV2D(dV.x, dV.y, 0.0f);
				float dVlen = dV2D.Length();

				if (dVlen > 1e-6f && bgL > 1e-6f) {
					Vec dVhat = dV2D / dVlen;
					float dot_goal = std::max(-1.0f, std::min(1.0f, dVhat.Dot(dirBallToGoal2D)));

					// Esponenziale più aggressivo
					float e_pos = std::exp(lambda_goal * dot_goal);
					float e_min = std::exp(-lambda_goal);
					float e_max = std::exp(lambda_goal);
					float gain01 = (e_pos - e_min) / (e_max - e_min);

					float mag = clamp01(dVlen / std::max(1.0f, max_delta_v));

					// 🔧 ZERO reward se direzione negativa (era min_touch_factor)
					float dir_factor = (dot_goal > 0.0f) ? gain01 : 0.0f;

					float touch = touch_base * mag * dir_factor * field_mult;

					// Boost se tocco da sotto (SOLO se verso porta)
					if (dot_goal > 0.5f) {
						float vertical_gap = state.ball.pos.z - player.pos.z;
						if (vertical_gap > under_ball_min_gap && dV.z > 0.0f) {
							float gap01 = clamp01((vertical_gap - under_ball_min_gap) /
								std::max(1.0f, under_ball_max_gap - under_ball_min_gap));
							float touch_boost = 1.0f + under_touch_boost * gap01;
							touch *= touch_boost;
						}
					}

					reward += touch;

					// 🆕 Bonus extra per shot perfetti
					if (dot_goal > 0.9f && mag > 0.6f) {
						reward += 5.0f * field_mult;
					}
				}
			}

			// ========================================================================
			// 🆕 RIMOSSO: Secondo "under ball bonus" continuo
			// Era confuso e controproducente - ora solo nel touch bonus
			// ========================================================================

			// ========================================================================
			// AIR ROLL BONUS (condizionato + ridotto)
			// ========================================================================
			if (vlen > airroll_speed_threshold && align_vel_desired > min_goal_push_dot) {
				Vec forward = player.rotMat.forward;
				float roll_rate_signed = player.angVel.Dot(forward);
				float roll_rate_abs = std::fabs(roll_rate_signed);

				float smooth_factor = (roll_rate_abs > min_airroll_rate)
					? smooth_factor_active
					: smooth_factor_inactive;
				ar_state.smoothed_roll_rate = smooth_factor * ar_state.smoothed_roll_rate +
					(1.0f - smooth_factor) * roll_rate_abs;

				float direction_threshold = min_airroll_rate * direction_detect_mult;
				float curr_direction = 0.0f;
				if (roll_rate_signed > direction_threshold) curr_direction = 1.0f;
				if (roll_rate_signed < -direction_threshold) curr_direction = -1.0f;

				bool direction_changed = false;
				if (curr_direction != 0.0f && ar_state.prev_roll_direction != 0.0f) {
					if (curr_direction * ar_state.prev_roll_direction < 0) {
						direction_changed = true;
					}
				}

				if (ar_state.smoothed_roll_rate > min_airroll_rate && !direction_changed) {
					ar_state.consecutive_ticks++;
					ar_state.stable_direction_ticks++;
					ar_state.total_airroll_time += 1.0f;

					if (ar_state.stable_direction_ticks >= min_ticks_for_bonus) {
						ar_state.has_reached_min_ticks = true;
					}

					if (ar_state.has_reached_min_ticks) {
						float progression = std::min(
							static_cast<float>(ar_state.stable_direction_ticks) /
							static_cast<float>(max_progression_ticks),
							1.0f
						);

						// 🔧 USA airroll_bonus_max = 1.2 (ridotto!)
						float current_bonus = airroll_bonus_base +
							(airroll_bonus_max - airroll_bonus_base) * progression;

						// Moltiplicatori qualità
						float intensity_factor = clamp01(ar_state.smoothed_roll_rate / 1.0f);
						intensity_factor = std::max(0.5f, intensity_factor);

						float speed_factor = clamp01(vlen / 1800.0f);
						speed_factor = 0.6f + 0.4f * speed_factor;

						float proximity_factor = clamp01(proximity * 2.0f);
						proximity_factor = 0.7f + 0.3f * proximity_factor;

						float consistency_factor = 1.0f;
						if (ar_state.stable_direction_ticks > consistency_start_ticks) {
							float consistency_progress = std::min(
								static_cast<float>(ar_state.stable_direction_ticks - consistency_start_ticks) / 80.0f,
								1.0f
							);
							consistency_factor = 1.0f + consistency_bonus_rate * consistency_progress;
						}

						// 🔧 Rinforza se allineato a desired2D
						float desired_factor = 0.7f + 0.3f * align_vel_desired;

						float avg_quality = (intensity_factor + speed_factor + proximity_factor) / 3.0f;
						float quality_mult = (avg_quality > quality_threshold) ? quality_multiplier : 1.0f;

						float airroll_bonus = current_bonus *
							intensity_factor *
							speed_factor *
							proximity_factor *
							consistency_factor *
							desired_factor *
							quality_mult *
							field_mult;

						ar_state.accumulated_bonus += airroll_bonus;
						reward += airroll_bonus;

						// Milestone bonus (ridotti)
						if (ar_state.stable_direction_ticks == 60 ||
							ar_state.stable_direction_ticks == 120) {
							reward += 0.5f * field_mult;  // Era 0.8
						}
					}
				}
				else {
					// Decay o cambio direzione
					if (direction_changed) {
						ar_state.consecutive_ticks = 0;
						ar_state.stable_direction_ticks = 0;
						ar_state.accumulated_bonus = 0.0f;
						ar_state.has_reached_min_ticks = false;
					}
					else if (ar_state.smoothed_roll_rate <= min_airroll_rate) {
						ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - decay_rate_active);
						ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_active);

						if (ar_state.stable_direction_ticks < min_ticks_for_bonus) {
							ar_state.has_reached_min_ticks = false;
						}
					}
				}

				if (curr_direction != 0.0f) {
					ar_state.prev_roll_direction = curr_direction;
				}

			}
			else {
				// Velocità troppo bassa o non verso porta → decay
				ar_state.consecutive_ticks = std::max(0, ar_state.consecutive_ticks - 1);
				ar_state.stable_direction_ticks = std::max(0, ar_state.stable_direction_ticks - decay_rate_slow);
				ar_state.smoothed_roll_rate *= 0.75f;

				if (ar_state.stable_direction_ticks < min_ticks_for_bonus) {
					ar_state.has_reached_min_ticks = false;
				}
			}

			return reward;
		}
	};

	class VelocityPlayerToNearestPlayerReward : public Reward {
	private:
		bool useScalarProjection;

	public:
		VelocityPlayerToNearestPlayerReward(bool use_scalar_projection = false)
			: useScalarProjection(use_scalar_projection) {
		}

		virtual void Reset(const GameState& initialState) override {
			// No internal state to reset
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			Vec vel = player.vel;
			Team playerTeam = player.team;

			Vec nearestOpponentPos;
			float minDistanceSq = 1000000.0f; // Large initial value
			bool foundOpponent = false;

			// Find nearest opponent
			for (const Player& otherPlayer : state.players) {
				if (otherPlayer.team == playerTeam || otherPlayer.carId == player.carId) {
					continue; // Skip teammates and self
				}

				Vec posDiff = otherPlayer.pos - player.pos;
				float distSq = posDiff.LengthSq();

				if (distSq < minDistanceSq) {
					minDistanceSq = distSq;
					nearestOpponentPos = otherPlayer.pos;
					foundOpponent = true;
				}
			}

			if (!foundOpponent) {
				return 0.0f;
			}

			Vec posDiff = nearestOpponentPos - player.pos;

			if (useScalarProjection) {
				// Vector version of v=d/t <=> t=d/v <=> 1/t=v/d
				// Max value should be max_speed / ball_radius = 2300 / 92.75 = 24.8
				// Used to guide agent towards nearest opponent
				float posDiffLength = posDiff.Length();
				if (posDiffLength == 0.0f) {
					return 0.0f;
				}
				Vec normalizedPosDiff = posDiff / posDiffLength;
				return vel.Dot(normalizedPosDiff) / CommonValues::CAR_MAX_SPEED;
			}
			else {
				// Regular component velocity
				float posDiffLength = posDiff.Length();
				if (posDiffLength == 0.0f) {
					return 0.0f;
				}
				Vec normalizedPosDiff = posDiff / posDiffLength;
				Vec normalizedVel = vel / CommonValues::CAR_MAX_SPEED;
				return normalizedPosDiff.Dot(normalizedVel);
			}
		}
	};

	class FacePlayerReward : public Reward {
	private:
		float supersonicBonus;
		float bumpReward;
		float demoReward;
		float goalConcedePenalty;
		float randomMovementPenalty;
		float prevFacingValue = 0.0f;
		bool wasSupersonic = false;

	public:
		FacePlayerReward(
			float supersonicBonus = 0.5f,
			float bumpReward = 1.0f,
			float demoReward = 2.0f,
			float goalConcedePenalty = -3.0f,
			float randomMovementPenalty = -0.1f
		) : supersonicBonus(supersonicBonus), bumpReward(bumpReward),
			demoReward(demoReward), goalConcedePenalty(goalConcedePenalty),
			randomMovementPenalty(randomMovementPenalty) {
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			float totalReward = 0.0f;

			// Find nearest opponent
			Vec nearestOpponentPos = Vec(0, 0, 0);
			float minDist = FLT_MAX;
			bool foundOpponent = false;

			for (const Player& opponent : state.players) {
				if (opponent.team == player.team || opponent.carId == player.carId) {
					continue;
				}

				foundOpponent = true;
				float opponentDist = (opponent.pos - player.pos).Length();

				if (opponentDist < minDist) {
					minDist = opponentDist;
					nearestOpponentPos = opponent.pos;
				}
			}

			if (!foundOpponent) {
				return 0.0f;
			}

			// Base continuous reward: facing towards opponent
			Vec dirToOpponent = (nearestOpponentPos - player.pos).Normalized();
			float facingValue = player.rotMat.forward.Dot(dirToOpponent);
			totalReward += facingValue;

			// Bonus: going towards opponent at supersonic speed
			bool isSupersonic = player.vel.Length() >= SUPERSONIC_THRESHOLD;
			float speedAlignment = (player.vel.Normalized()).Dot(dirToOpponent);

			if (isSupersonic && speedAlignment > 0.7f) { // Moving significantly towards opponent
				totalReward += supersonicBonus * speedAlignment;
			}

			// Penalty: random movement (facing direction changes too much without reason)
			if (!player.prev) {
				prevFacingValue = facingValue;
				wasSupersonic = isSupersonic;
			}
			else {
				float facingChange = abs(facingValue - prevFacingValue);
				bool wasSupersonicPrev = wasSupersonic;
				wasSupersonic = isSupersonic;

				// Penalize rapid facing changes when not at supersonic (indicative of random movement)
				if (facingChange > 0.5f && !isSupersonic && !wasSupersonicPrev) {
					totalReward += randomMovementPenalty;
				}
				prevFacingValue = facingValue;
			}

			// Event-based rewards
			if (player.eventState.bump) {
				totalReward += bumpReward;
			}

			if (player.eventState.demo) {
				totalReward += demoReward;
			}

			// Strict penalty: conceding goal while actively facing/chasing opponent
			if (state.goalScored) {
				bool scoredOnMe = (player.team == RS_TEAM_FROM_Y(state.ball.pos.y));
				if (scoredOnMe && facingValue > 0.3f) { // Was actively facing opponent
					totalReward += goalConcedePenalty;
				}
			}

			return totalReward;
		}

		virtual void Reset(const GameState& initialState) override {
			prevFacingValue = 0.0f;
			wasSupersonic = false;
		}
	};

	class StrongTouchReward : public Reward {
	public:
		float minRewardedVel, maxRewardedVel;
		StrongTouchReward(float minSpeedKPH = 20, float maxSpeedKPH = 130) {
			minRewardedVel = RLGC::Math::KPHToVel(minSpeedKPH);
			maxRewardedVel = RLGC::Math::KPHToVel(maxSpeedKPH);
		}

		virtual float GetReward(const Player& player, const GameState& state, bool isFinal) override {
			if (!state.prev)
				return 0;

			if (player.ballTouchedStep) {
				float hitForce = (state.ball.vel - state.prev->ball.vel).Length();
				if (hitForce < minRewardedVel)
					return 0;

				return RS_MIN(1, hitForce / maxRewardedVel);
			}
			else {
				return 0;
			}
		}
	};
}
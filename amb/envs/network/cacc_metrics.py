"""Read-only CACC safety telemetry; TTC is a longitudinal gap proxy."""

import math
import numpy as np


def get_cacc_infos(env, was_collided, actions=None):
    leaders = np.concatenate(([env.v0s[env.t]], env.vs_cur[:-1]))
    actions = np.asarray(actions).reshape(-1) if actions is not None else None
    infos = []
    for agent_id in range(env.n_agent):
        gap = float(env.hs_cur[agent_id])
        speed = float(env.vs_cur[agent_id])
        leader_speed = float(leaders[agent_id])
        closing_speed = speed - leader_speed
        ttc = gap / closing_speed if gap > 0 and closing_speed > 0 else None
        infos.append({"cacc": {
            "episode_seed": env.episode_seed,
            "step": env.t,
            "time_s": env.t * env.dt,
            "agent_id": agent_id,
            "headway_m": gap,
            "headway_min_m": env.h_min,
            "speed_mps": speed,
            "leader_speed_mps": leader_speed,
            "acceleration_mps2": float(env.us_cur[agent_id]),
            "headway_error_m": gap - env.h_star,
            "speed_error_mps": speed - env.v_star,
            "requested_action": int(actions[agent_id]) if actions is not None else None,
            "ttc_proxy_s": ttc,
            "headway_violation": bool(gap < env.h_min),
            "collision": bool(env.collision),
            "first_collision": bool(env.collision and not was_collided),
            "dynamics_advanced": not was_collided,
        }})
    return infos


class CACCEpisodeMetrics:
    def __init__(self, ttc_threshold_s=1.0):
        if not math.isfinite(ttc_threshold_s) or ttc_threshold_s <= 0:
            raise ValueError("ttc_threshold_s must be finite and positive")
        self.threshold = ttc_threshold_s
        self.seed = None
        self.steps = 0
        self.team_return = 0.0
        self.collision = False
        self.first_collision_time = None
        self.min_gap = None
        self.min_ttc = None
        self.ttc_valid = 0
        self.ttc_below = 0
        self.active_samples = 0
        self.headway_error_sq = 0.0
        self.speed_error_sq = 0.0
        self.action_counts = None

    def update(self, infos, rewards):
        rows = [info["cacc"] for info in infos]
        if not rows:
            raise ValueError("CACC telemetry is empty")
        seed = rows[0]["episode_seed"]
        if self.steps and seed != self.seed:
            raise ValueError("Cannot combine different CACC episodes")
        self.seed = seed
        self.steps += 1
        # NetworkEnv repeats the same team reward for every agent.
        self.team_return += float(np.mean(rewards))
        if self.action_counts is None:
            self.action_counts = [[0] * 4 for _ in rows]
        for row in rows:
            self.collision |= row["collision"]
            if row["first_collision"] and self.first_collision_time is None:
                self.first_collision_time = row["time_s"]
            if not row["dynamics_advanced"]:
                continue  # frozen post-collision states are not fresh samples
            self.active_samples += 1
            self.headway_error_sq += row["headway_error_m"] ** 2
            self.speed_error_sq += row["speed_error_mps"] ** 2
            if row["requested_action"] is not None:
                self.action_counts[row["agent_id"]][row["requested_action"]] += 1
            gap = row["headway_m"]
            self.min_gap = gap if self.min_gap is None else min(self.min_gap, gap)
            ttc = row["ttc_proxy_s"]
            if ttc is not None:
                self.min_ttc = ttc if self.min_ttc is None else min(self.min_ttc, ttc)
                self.ttc_valid += 1
                self.ttc_below += int(ttc < self.threshold)

    def result(self):
        return {
            "schema_version": 2, "episode_seed": self.seed,
            "steps": self.steps, "team_return": self.team_return,
            "collision": self.collision,
            "first_collision_time_s": self.first_collision_time,
            "min_headway_m": self.min_gap, "min_ttc_proxy_s": self.min_ttc,
            "ttc_threshold_s": self.threshold,
            "active_agent_samples": self.active_samples,
            "headway_rmse_m": math.sqrt(self.headway_error_sq / self.active_samples) if self.active_samples else None,
            "speed_rmse_mps": math.sqrt(self.speed_error_sq / self.active_samples) if self.active_samples else None,
            "action_counts_by_agent": self.action_counts,
            "ttc_valid_samples": self.ttc_valid,
            "ttc_below_threshold_samples": self.ttc_below,
            "ttc_below_threshold_fraction": (
                self.ttc_below / self.ttc_valid if self.ttc_valid else None
            ),
        }

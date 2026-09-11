"""Bounded clean CACC pilot with fixed validation seeds and checkpoint selection."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from amb.runners.single.on_policy_runner import OnPolicyRunner
from amb.utils.run_manifest import checkpoint_inventory, finish_run_manifest, require_clean_source


def select_candidate(records, selection="final"):
    initial = records[0]
    candidates = records[1:]
    if not candidates:
        raise ValueError("The pilot did not evaluate a trained checkpoint")
    final = max(candidates, key=lambda row: row["checkpoint_step"])
    best = max(candidates, key=lambda row: row["mean_team_return"])
    if selection not in ("final", "best-validation-return"):
        raise ValueError("Unknown checkpoint selection rule")
    selected = final if selection == "final" else best
    late = candidates[len(candidates) // 2:]
    x = np.array([row["checkpoint_step"] for row in late], dtype=float)
    y = np.array([row["mean_team_return"] for row in late], dtype=float)
    slope = float(np.polyfit((x - x[0]) / 10000, y, 1)[0]) if len(late) >= 2 else None
    return {
        "checkpoint_step": selected["checkpoint_step"], "selection_rule": selection,
        "final_checkpoint_step": final["checkpoint_step"],
        "best_validation_checkpoint_step": best["checkpoint_step"],
        "final_return_change_from_initial": final["mean_team_return"] - initial["mean_team_return"],
        "late_curve": {"checkpoint_steps": x.astype(int).tolist(),
                       "return_range": float(np.ptp(y)), "return_slope_per_10000_steps": slope},
        "convergence_status": "requires_curve_review",
    }


class CleanPilotRunner(OnPolicyRunner):
    def eval(self):
        protocol = self.algo_args["extra"]
        # One worker makes the evaluated seed set independent of episode lengths.
        self.eval_envs.envs[0].seed(protocol["validation_seed_start"])
        rng_states = random.getstate(), np.random.get_state(), torch.get_rng_state()
        try:
            super().eval()
        finally:
            random.setstate(rng_states[0])
            np.random.set_state(rng_states[1])
            torch.set_rng_state(rng_states[2])
        episodes = deepcopy(self.logger.cacc_episodes)
        expected = list(range(protocol["validation_seed_start"], protocol["validation_seed_start"] + self.algo_args["train"]["eval_episodes"]))
        if [row["episode_seed"] for row in episodes] != expected:
            raise RuntimeError("Validation episodes did not use the declared fixed seeds")
        record = {
            "checkpoint_step": self.logger.timestep,
            "mean_team_return": float(np.mean([row["team_return"] for row in episodes])),
            "std_team_return": float(np.std([row["team_return"] for row in episodes], ddof=1)) if len(episodes) > 1 else None,
            "collision_count": sum(row["collision"] for row in episodes),
            "episode_count": len(episodes),
            "mean_headway_rmse_m": float(np.mean([row["headway_rmse_m"] for row in episodes])),
            "mean_speed_rmse_mps": float(np.mean([row["speed_rmse_mps"] for row in episodes])),
            "episodes": episodes,
        }
        with (Path(self.run_dir) / "pilot_validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(f"Pilot validation: step={record['checkpoint_step']}, return={record['mean_team_return']:.2f}, "
              f"collisions={record['collision_count']}/{len(episodes)}, gap_rmse={record['mean_headway_rmse_m']:.3f} m", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=["catchup", "slowdown"], default="catchup")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--eval-every-steps", type=int, default=20000)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--validation-seed-start", type=int, default=1000000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--checkpoint-selection", choices=["final", "best-validation-return"], default="final")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/cacc_clean_pilot")
    options = parser.parse_args()
    batch = options.workers * options.rollout_length
    if min(options.workers, options.rollout_length, options.steps, options.eval_every_steps, options.validation_episodes) <= 0:
        parser.error("budgets and worker counts must be positive")
    if options.steps % batch or options.eval_every_steps % batch or options.steps % options.eval_every_steps:
        parser.error("steps must be a multiple of the evaluation interval; both must be whole rollout batches")
    max_seed = 2**32 - 1
    if options.seed < 0 or options.seed * 50000 > max_seed:
        parser.error("seed must fit the runner's initial evaluation seed range")
    # Conservative bounds allow even one-step episodes without entering validation seeds.
    train_end = options.seed + (options.workers - 1) * 1000 + options.steps // options.workers + 1
    if options.validation_seed_start <= train_end or options.validation_seed_start + options.validation_episodes > max_seed:
        parser.error("validation seeds must follow all possible training seeds and fit uint32")
    source = require_clean_source()
    config = json.loads((ROOT / f"experiment/settings/network/{options.scenario}/mappo.json").read_text())
    train = config["algo_args"]["train"]
    train.update({
        "cuda": False, "torch_threads": 1, "seed": options.seed, "seed_specify": True,
        "num_env_steps": options.steps, "episode_length": options.rollout_length,
        "n_rollout_threads": options.workers, "n_eval_rollout_threads": 1,
        "use_eval": True, "eval_episodes": options.validation_episodes,
        "eval_interval": options.eval_every_steps // batch, "log_interval": 1,
        "slice": True, "slice_interval": 1, "model_dir": None,
        "log_dir": str(options.output_dir.resolve()),
    })
    config["env_args"]["record_legacy"] = False
    config["main_args"]["exp_name"] = "stage1-clean-pilot"
    config["algo_args"]["extra"] = {
        "purpose": "pilot", "validation_seed_start": options.validation_seed_start,
        "eval_seed_override": list(range(options.validation_seed_start, options.validation_seed_start + options.validation_episodes)),
        "validation_protocol": "fixed consecutive seeds in one worker; preserve process RNG around evaluation",
        "checkpoint_selection": options.checkpoint_selection,
        "selection_rule": "final checkpoint by default; also report best mean validation return without collision filtering",
        "convergence_protocol": "review return curves, late trend and variation; no zero-collision or fixed percentage gates",
        "terminal_semantics": "native CACC finite episode; reward and delayed collision termination unchanged",
    }
    runner = CleanPilotRunner(config["main_args"], config["algo_args"], config["env_args"])
    error = None
    try:
        runner.save_slice(0)
        runner.run()
        records = [json.loads(line) for line in (Path(runner.run_dir) / "pilot_validation.jsonl").read_text().splitlines()]
        selection = select_candidate(records, options.checkpoint_selection)
        selected_dir = Path(runner.run_dir) / "slice" / str(selection["checkpoint_step"])
        selection.update({"source": source, "purpose": "pilot", "checkpoint_dir": str(selected_dir.resolve()),
                          "validation_evaluations": len(records), "checkpoint": checkpoint_inventory(selected_dir, include_lineage=False)})
        selection["checkpoint_alternatives"] = {
            label: checkpoint_inventory(Path(runner.run_dir) / "slice" / str(selection[key]), include_lineage=False)
            for label, key in (("final", "final_checkpoint_step"), ("best_validation", "best_validation_checkpoint_step"))
        }
        (Path(runner.run_dir) / "pilot_summary.json").write_text(json.dumps(selection, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Pilot summary: {Path(runner.run_dir).resolve() / 'pilot_summary.json'}", flush=True)
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            runner.close()
        except BaseException as exc:
            error = error or {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            finish_run_manifest(runner.run_dir, error)


if __name__ == "__main__":
    main()

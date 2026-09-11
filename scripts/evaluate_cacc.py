"""Paired native CACC evaluation of a saved MAPPO victim.

Uses the existing AMB runner/IGS. No new attack algorithm is introduced.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml

from amb.runners.perturbation.base_runner import BaseRunner
from amb.utils.env_utils import set_seed
from amb.utils.run_manifest import finish_run_manifest, require_clean_source, source_state


class BudgetAudit:
    """Check all candidates; count applied agent-steps for the all-time mask."""

    def __init__(self, attack, n_agents, selected_ids, epsilon):
        self.attack = attack
        self.n_agents = n_agents
        self.selected_ids = selected_ids
        self.epsilon = epsilon
        self.calls = 0
        self.applied_agent_steps = 0
        self.max_linf = 0.0

    def perturb(self, agent, obs, *args):
        result = self.attack.perturb(agent, obs, *args)
        values = result.detach().cpu().numpy()
        norm = float(np.max(np.abs(values - obs)))
        if not np.isfinite(values).all() or norm > self.epsilon + 1e-6:
            raise RuntimeError("Observation attack exceeded its finite L-infinity budget")
        if self.calls % self.n_agents in self.selected_ids:
            self.max_linf = max(self.max_linf, norm)
            self.applied_agent_steps += len(obs)
        self.calls += 1
        return result


def evaluate_condition(options, victim_config, output, condition):
    victim = deepcopy(victim_config["algo_args"]["train"])
    victim["model_dir"] = str(options.victim_dir / "models")
    env_args = deepcopy(victim_config["env_args"])
    train = yaml.safe_load((ROOT / "amb/configs/algos_cfgs/mappo_traitor.yaml").read_text())
    epsilon = options.epsilon if condition in ("gaussian", "igs") else 0.0
    iterations = options.iterations if condition == "igs" else 0
    # The mask covers the unchanged native 600-step horizon.
    train.update({
        "cuda": False, "torch_threads": 1, "seed": options.attack_seed,
        "seed_specify": True, "n_rollout_threads": 1, "n_eval_rollout_threads": 1,
        "num_env_steps": 0, "episode_length": 600, "eval_episodes": 1,
        "use_eval": True, "log_dir": str(output), "model_dir": None,
        "perturb_epsilon": epsilon, "perturb_iters": iterations,
        "adaptive_alpha": True, "targeted_attack": False,
        "perturbation_eps": 0.0, "traitor_eps": 0.0,
        "adv_all": False, "adv_agent_ids": options.agent_ids,
        "perturb_timesteps": [True] * 600, "slice": False,
    })
    extra = {
        "purpose": options.purpose, "condition": condition,
        "eval_seed_override": options.eval_seeds,
        "attack_seed_rule": "attack_seed + evaluation_seed_index, reset before each episode",
        "budget_units": "normalized observation L-infinity per selected agent per step",
        "evaluation_protocol": "one worker, paired explicit episode seeds, deterministic victim",
    }
    runner = BaseRunner(
        {"env": "network", "algo": "mappo", "victim": "mappo", "run": "perturbation", "exp_name": condition},
        {"train": train, "victim": victim, "extra": extra}, env_args,
    )
    rows = []
    error = None
    try:
        if runner.eval_envs.envs[0].env.env.T != 600:
            raise ValueError("This protocol requires the native 600-step CACC horizon")
        if any(i < 0 or i >= runner.num_agents for i in options.agent_ids):
            raise ValueError("agent_ids are outside this victim's agent range")
        runner.restore()
        runner.logger.init()
        runner.logger.episode_init(0)
        original_attack = runner.attack
        for index, seed in enumerate(options.eval_seeds):
            attack_seed = options.attack_seed + index
            set_seed({"seed_specify": True, "seed": attack_seed})
            runner.eval_envs.envs[0].seed(seed)
            audit = BudgetAudit(original_attack, runner.num_agents, options.agent_ids, epsilon)
            runner.attack = audit
            if condition == "clean":
                runner.eval()
            else:
                runner.eval_adv()
            record = deepcopy(runner.logger.cacc_episodes[0])
            record.update({
                "condition": condition, "attack_seed": attack_seed,
                "epsilon": epsilon, "iterations": iterations,
                "agent_ids": options.agent_ids, "observed_linf_max": audit.max_linf,
                "applied_agent_steps": audit.applied_agent_steps,
                "candidate_calls": audit.calls,
            })
            rows.append(record)
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
    return {"run_dir": str(Path(runner.run_dir).resolve()), "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--victim-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/cacc_evaluation")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[500, 501])
    parser.add_argument("--attack-seed", type=int, default=1)
    parser.add_argument("--agent-ids", nargs="+", type=int, default=[0])
    parser.add_argument("--attacks", nargs="+", choices=["clean", "zero", "gaussian", "igs"], default=["clean", "zero", "gaussian", "igs"])
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--purpose", choices=["smoke", "pilot", "baseline"], default="pilot")
    parser.add_argument("--allow-dirty", action="store_true", help="Exploratory runs only; snapshots are saved")
    options = parser.parse_args()
    if not math.isfinite(options.epsilon) or options.epsilon < 0 or options.iterations < 1:
        parser.error("epsilon must be finite and nonnegative; iterations must be positive")
    if len(set(options.agent_ids)) != len(options.agent_ids) or len(set(options.attacks)) != len(options.attacks):
        parser.error("agent IDs and attack conditions must be unique")
    if len(set(options.eval_seeds)) != len(options.eval_seeds):
        parser.error("evaluation seeds must be unique")
    if any(s < 0 or s >= 2**32 - 1 for s in options.eval_seeds):
        parser.error("evaluation seeds must allow at least one subsequent reset in uint32 range")
    if options.attack_seed < 0 or options.attack_seed + len(options.eval_seeds) >= 2**32:
        parser.error("attack seeds must fit the NumPy seed range")
    if options.attack_seed * 50000 >= 2**32:
        parser.error("attack_seed * 50000 must fit the runner's initial evaluation seed range")
    options.victim_dir = options.victim_dir.resolve()
    config = json.loads((options.victim_dir / "config.json").read_text(encoding="utf-8"))
    if config["main_args"]["env"] != "network" or config["main_args"]["algo"] != "mappo" or config["env_args"]["scenario"] not in ("catchup", "slowdown"):
        parser.error("this evaluator supports native CACC MAPPO victims")
    if not (options.victim_dir / "models").is_dir():
        parser.error("victim models directory does not exist")
    source = source_state() if options.allow_dirty else require_clean_source()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    output = options.output_dir.resolve() / f"{stamp}-{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    results = {condition: evaluate_condition(options, config, output, condition) for condition in options.attacks}
    zero_equal = None
    if "clean" in results and "zero" in results:
        fields = ("team_return", "steps", "collision", "min_headway_m", "min_ttc_proxy_s", "ttc_valid_samples", "ttc_below_threshold_samples")
        zero_equal = all(
            all(clean[key] == zero[key] for key in fields)
            for clean, zero in zip(results["clean"]["episodes"], results["zero"]["episodes"])
        )
        if not zero_equal:
            raise RuntimeError("Zero-budget evaluation differs from paired clean evaluation")
    summary = {
        "source": source, "purpose": options.purpose, "scenario": config["env_args"]["scenario"],
        "victim_dir": str(options.victim_dir), "zero_matches_clean": zero_equal, "conditions": results,
    }
    path = output / "summary.json"
    path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Paired evaluation summary: {path}", flush=True)


if __name__ == "__main__":
    main()

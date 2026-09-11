import json
from pathlib import Path

import pytest


@pytest.fixture
def make_cacc_runner(tmp_path):
    """Small real MAPPO runner; each test owns and closes its environments."""
    from amb.runners.single.on_policy_runner import OnPolicyRunner
    from amb.utils.run_manifest import finish_run_manifest

    runners = []

    def make(**overrides):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "experiment/settings/network/catchup/mappo.json").read_text())
        train = config["algo_args"]["train"]
        train.update({
            "cuda": False, "torch_threads": 1, "hidden_sizes": [16, 16],
            "ppo_epoch": 1, "critic_epoch": 1, "n_rollout_threads": 1,
            "n_eval_rollout_threads": 1, "episode_length": 4, "num_env_steps": 8,
            "use_eval": False, "eval_interval": 25, "log_interval": 1000,
            "log_dir": str(tmp_path), "seed": 7, "slice": False,
            **overrides,
        })
        config["main_args"]["exp_name"] = "test_training"
        runner = OnPolicyRunner(config["main_args"], config["algo_args"], config["env_args"])
        runners.append(runner)
        return runner

    yield make
    for runner in runners:
        runner.close()
        finish_run_manifest(runner.run_dir)

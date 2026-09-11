"""Exercise the real PPO observation-attack runner over complete short episodes."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from amb.runners.perturbation.base_runner import BaseRunner


@pytest.mark.parametrize("epsilon,iterations", [(0.0, 0), (0.05, 1)])
def test_discrete_attack_evaluation_and_zero_budget_replay(tmp_path, epsilon, iterations):
    root = Path(__file__).resolve().parents[1]
    configs = root / "amb/configs/algos_cfgs"
    victim = yaml.safe_load((configs / "mappo.yaml").read_text())
    train = yaml.safe_load((configs / "mappo_traitor.yaml").read_text())
    common = {
        "hidden_sizes": [16, 16], "cuda": False, "torch_threads": 1,
        "n_rollout_threads": 1, "n_eval_rollout_threads": 1,
        "episode_length": 60, "eval_episodes": 1, "use_eval": True,
        "num_env_steps": 0, "model_dir": None, "seed": 7,
    }
    victim.update(common)
    train.update(common)
    train.update({
        "log_dir": str(tmp_path), "perturb_epsilon": epsilon,
        "perturb_iters": iterations, "perturbation_eps": 0.0,
        "perturb_timesteps": [True] * 60,
        "targeted_attack": False, "adv_all": False, "adv_agent_ids": [0],
    })
    env_args = {
        "scenario": "catchup", "network_cfg": "config_ma2c_nc_catchup.ini",
        "override_args": {"episode_length_sec": 6},
    }
    runner = BaseRunner(
        {"env": "network", "algo": "mappo", "victim": "mappo",
         "run": "perturbation", "exp_name": "test_discrete"},
        {"train": train, "victim": victim, "extra": {}}, env_args,
    )
    divergences = []
    runner.logger.eval_log_Rpi = lambda values: divergences.append(values)
    try:
        runner.logger.init()
        runner.logger.episode_init(0)
        runner.eval_envs.envs[0].seed(500)
        runner.eval()
        clean_returns = deepcopy(runner.logger.eval_episode_rewards)
        runner.eval_envs.envs[0].seed(500)
        runner.eval_adv()
        assert len(divergences) == 60
        for row in divergences:
            assert set(row) == {f"agent_{i}" for i in range(8)}
            assert np.isfinite(list(row.values())).all()
        if epsilon == 0:
            np.testing.assert_array_equal(runner.logger.eval_episode_rewards, clean_returns)
            np.testing.assert_allclose([list(row.values()) for row in divergences], 0, atol=1e-7)
        else:
            # Only agent 0 was attacked; the other distributions must be unchanged.
            np.testing.assert_allclose(
                [[row[f"agent_{i}"] for i in range(1, 8)] for row in divergences],
                0, atol=1e-7,
            )
    finally:
        runner.close()

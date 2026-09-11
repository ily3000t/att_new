import random

import numpy as np
import torch

from scripts.train_cacc_pilot import CleanPilotRunner, select_candidate


def test_pilot_keeps_final_and_best_return_without_safety_filters():
    initial = {"checkpoint_step": 0, "mean_team_return": -100, "collision_count": 0, "mean_headway_rmse_m": 10}
    unsafe = {"checkpoint_step": 20, "mean_team_return": -10, "collision_count": 1, "mean_headway_rmse_m": 3}
    safe = {"checkpoint_step": 40, "mean_team_return": -60, "collision_count": 0, "mean_headway_rmse_m": 6}
    selected = select_candidate([initial, unsafe, safe])
    assert selected["checkpoint_step"] == 40
    assert selected["best_validation_checkpoint_step"] == 20
    assert select_candidate([initial, unsafe, safe], "best-validation-return")["checkpoint_step"] == 20
    assert select_candidate([initial, unsafe])["checkpoint_step"] == 20
    unchanged = {**initial, "checkpoint_step": 40}
    assert select_candidate([initial, unchanged])["checkpoint_step"] == 40


def test_pilot_uses_absolute_change_when_initial_return_is_zero():
    records = [{"checkpoint_step": step, "mean_team_return": reward, "collision_count": 2}
               for step, reward in [(0, 0), (10000, -10), (20000, -5), (30000, -5)]]
    result = select_candidate(records)
    assert result["checkpoint_step"] == 30000
    assert result["final_return_change_from_initial"] == -5
    assert result["late_curve"]["return_range"] == 0
    assert abs(result["late_curve"]["return_slope_per_10000_steps"]) < 1e-10
    assert result["convergence_status"] == "requires_curve_review"


def test_pilot_repeats_validation_seeds_without_changing_training_rng(make_cacc_runner):
    runner = make_cacc_runner(use_eval=True, eval_episodes=2, runner_cls=CleanPilotRunner)
    runner.algo_args["extra"] = {"validation_seed_start": 1000000}
    runner.logger.init()
    runner.logger.episode_init(0)
    before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    runner.eval()
    first = runner.logger.cacc_episodes.copy()
    runner.eval()
    assert runner.logger.cacc_episodes == first
    assert [row["episode_seed"] for row in first] == [1000000, 1000001]
    after = random.getstate(), np.random.get_state(), torch.get_rng_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1][1], after[1][1])
    assert before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])

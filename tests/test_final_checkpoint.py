from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("rollouts,interval,expected_saves", [
    (2, 25, [8]), (3, 2, [8, 12]), (2, 2, [8]),
])
def test_final_checkpoint_contains_last_update(make_cacc_runner, rollouts, interval, expected_saves):
    runner = make_cacc_runner(num_env_steps=4 * rollouts, eval_interval=interval)
    saved_at = []
    original_save = runner.save

    def save():
        saved_at.append(runner.current_timestep)
        original_save()

    runner.save = save
    runner.run()
    assert saved_at == expected_saves
    for agent_id, agent in enumerate(runner.agents):
        weights = torch.load(Path(runner.save_dir) / str(agent_id) / "actor.pth", weights_only=True)
        assert weights.keys() == agent.actor.state_dict().keys()
        for key, value in agent.actor.state_dict().items():
            assert torch.equal(weights[key], value)
    assert (Path(runner.save_dir) / "critic.pth").is_file()
    assert (Path(runner.save_dir) / "value_normalizer.pth").is_file()

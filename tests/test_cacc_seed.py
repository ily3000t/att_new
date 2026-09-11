import numpy as np
import pytest

from amb.envs.network.network_env import NetworkEnv


def make_env(scenario, seed):
    env = NetworkEnv({
        "scenario": scenario,
        "network_cfg": f"config_ma2c_nc_{scenario}.ini",
    })
    env.seed(seed)
    return env


@pytest.mark.parametrize("scenario", ["catchup", "slowdown"])
def test_seed_controls_first_episode_and_reseeding(scenario):
    env = make_env(scenario, 12345)
    core = env.env.env
    assert core.cur_episode == 0  # seeding does not silently consume a reset
    first = env.reset()[0]
    assert core.episode_seed == 12345
    draw = np.random.RandomState(12345).rand()
    if scenario == "catchup":
        assert core.hs_cur[0] == core.h_star * (1.5 + draw)
    else:
        assert core.vs_cur[0] == core.v_star * (1.5 + draw)
    second = env.reset()[0]
    assert core.episode_seed == 12346
    assert not np.array_equal(first, second)
    env.seed(12345)
    np.testing.assert_array_equal(env.reset()[0], first)


@pytest.mark.parametrize("scenario", ["catchup", "slowdown"])
def test_seeded_environments_replay_and_workers_are_distinct(scenario):
    left = make_env(scenario, 42)
    right = make_env(scenario, 42)
    other_worker = make_env(scenario, 1042)
    np.testing.assert_array_equal(left.reset()[0], right.reset()[0])
    assert not np.array_equal(left.env.state, other_worker.reset()[0])
    for step in range(600):
        actions = np.full((8, 1), 3)
        a = left.step(actions)
        # An interleaved worker must not alter either trajectory.
        if step % 11 == 0:
            other_worker.reset()
        b = right.step(actions)
        for idx in (0, 1, 2, 3):
            np.testing.assert_array_equal(a[idx], b[idx])
        if np.all(a[3]):
            break
    assert np.all(a[3])


@pytest.mark.parametrize("scenario", ["catchup", "slowdown"])
def test_environment_does_not_change_process_numpy_rng(scenario):
    before = np.random.get_state()
    env = make_env(scenario, 19)
    env.reset()
    env.reset()
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_action_space_samples_are_seeded():
    env = make_env("catchup", 31)
    samples = [[space.sample() for space in env.action_space] for _ in range(10)]
    env.seed(31)
    assert samples == [[space.sample() for space in env.action_space] for _ in range(10)]

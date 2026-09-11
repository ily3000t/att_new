import numpy as np
import pytest

from amb.data.episode_buffer import EpisodeBuffer
from amb.envs.network.network_env import NetworkEnv


class AffineNormalizer:
    def denormalize(self, values):
        return np.asarray(values) * 2 + 10


@pytest.mark.parametrize("proper_limits", [False, True])
@pytest.mark.parametrize("gae", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("terminal", [None, 0, 2])
def test_three_step_returns_with_two_workers(proper_limits, gae, normalized, terminal):
    scheme = {
        "rewards": {"vshape": (1,)},
        "returns": {"vshape": (1,), "extra": ["more_length"]},
        "value_preds": {"vshape": (1,), "extra": ["more_length"]},
        "masks": {"vshape": (1,), "offset": 1, "init_value": 1},
        "bad_masks": {"vshape": (1,), "offset": 1, "init_value": 1},
    }
    buffer = EpisodeBuffer({
        "episode_length": 3, "gamma": 0.5, "gae_lambda": 1,
        "use_gae": gae, "use_proper_time_limits": proper_limits,
    }, 2, scheme)
    buffer.data["rewards"][:, :, 0] = [[1, 2, 3], [11, 12, 13]]
    values = np.array([[4, 5, 6], [4, 5, 6]], dtype=np.float32)
    next_values = np.array([[10], [20]], dtype=np.float32)
    if normalized:
        values = (values - 10) / 2
        next_values = (next_values - 10) / 2
    buffer.data["value_preds"][:, :3, 0] = values
    if terminal is not None:
        buffer.data["masks"][:, terminal + 1] = 0
    buffer.compute_returns(next_values, AffineNormalizer() if normalized else None)
    expected = {
        None: [[4, 6, 8], [22.75, 23.5, 23]],
        0: [[1, 6, 8], [11, 23.5, 23]],
        2: [[2.75, 3.5, 3], [20.25, 18.5, 13]],
    }[terminal]
    np.testing.assert_allclose(buffer.data["returns"][:, :3, 0], expected)


def test_cacc_finite_horizon_remains_an_episode_end():
    env = NetworkEnv({
        "scenario": "catchup", "network_cfg": "config_ma2c_nc_catchup.ini",
        "override_args": {"episode_length_sec": 1},
    })
    env.seed(42)
    env.reset()
    for step in range(10):
        _, _, _, dones, infos, _ = env.step(np.full((8, 1), 3))
        assert bool(np.all(dones)) == (step == 9)
        assert all(not info.get("bad_transition", False) for info in infos)
        assert all(not info["cacc"]["collision"] for info in infos)
    env.close()

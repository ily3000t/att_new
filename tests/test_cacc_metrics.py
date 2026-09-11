import json
from types import SimpleNamespace

import numpy as np
import pytest

from amb.envs.network.cacc_metrics import CACCEpisodeMetrics, get_cacc_infos
from amb.envs.network.network_env import NetworkEnv


def test_ttc_closing_nonclosing_and_invalid_gap():
    core = SimpleNamespace(
        v0s=[10.0, 10.0], vs_cur=np.array([12.0, 11.0, 12.0]),
        hs_cur=np.array([4.0, 8.0, -1.0]), us_cur=np.zeros(3),
        n_agent=3, t=1, dt=0.1, h_min=1.0, episode_seed=10, collision=True,
    )
    rows = get_cacc_infos(core, False)
    assert rows[0]["cacc"]["ttc_proxy_s"] == 2.0
    assert rows[1]["cacc"]["ttc_proxy_s"] is None
    assert rows[2]["cacc"]["ttc_proxy_s"] is None
    assert rows[2]["cacc"]["headway_violation"]
    assert rows[0] is not rows[1]
    json.dumps(rows, allow_nan=False)


def test_frozen_collision_is_counted_once_and_team_reward_not_multiplied():
    core = SimpleNamespace(
        v0s=[10.0, 10.0], vs_cur=np.array([12.0]), hs_cur=np.array([0.5]),
        us_cur=np.zeros(1), n_agent=1, t=1, dt=0.1, h_min=1.0,
        episode_seed=10, collision=True,
    )
    metrics = CACCEpisodeMetrics()
    metrics.update(get_cacc_infos(core, False), [-1000] * 8)
    metrics.update(get_cacc_infos(core, True), [-1000] * 8)
    result = metrics.result()
    assert result["team_return"] == -2000
    assert result["steps"] == 2
    assert result["active_agent_samples"] == 1
    assert result["first_collision_time_s"] == 0.1
    assert result["ttc_valid_samples"] == 1
    assert result["ttc_below_threshold_fraction"] == 1.0


def test_safe_nonclosing_ttc_is_null_not_zero():
    env = NetworkEnv({"scenario": "catchup", "network_cfg": "config_ma2c_nc_catchup.ini"})
    env.seed(17)
    env.reset()
    transition = env.step(np.zeros((8, 1), dtype=int))
    metrics = CACCEpisodeMetrics()
    metrics.update(transition[4], transition[2])
    assert metrics.result()["min_ttc_proxy_s"] is None
    assert metrics.result()["ttc_below_threshold_fraction"] is None
    assert metrics.result()["team_return"] == transition[2][0]
    json.dumps(metrics.result(), allow_nan=False)


@pytest.mark.parametrize("threshold", [0, -1, float('nan'), float('inf')])
def test_invalid_ttc_threshold(threshold):
    with pytest.raises(ValueError):
        CACCEpisodeMetrics(threshold)

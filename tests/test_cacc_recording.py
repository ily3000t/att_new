import numpy as np
import pytest

from amb.envs.network.network_env import NetworkEnv


@pytest.mark.parametrize("scenario", ["catchup", "slowdown"])
def test_legacy_recording_is_opt_in_and_does_not_change_episodes(scenario):
    args = {"scenario": scenario, "network_cfg": f"config_ma2c_nc_{scenario}.ini"}
    normal = NetworkEnv(args)
    recorded = NetworkEnv({**args, "record_legacy": True})
    normal.seed(42)
    recorded.seed(42)
    try:
        for episode in range(3):
            np.testing.assert_array_equal(normal.reset()[0], recorded.reset()[0])
            for _ in range(600):
                left = normal.step(np.full((8, 1), 3))
                right = recorded.step(np.full((8, 1), 3))
                for index in range(4):
                    np.testing.assert_array_equal(left[index], right[index])
                assert left[4] == right[4]
                if np.all(left[3]):
                    break
            assert np.all(left[3])
            assert not normal.env.env.is_record
            assert not hasattr(normal.env.env, "control_data")
            assert not hasattr(normal.env.env, "traffic_data")
            assert len(normal.env.env.hs) <= 601  # only this episode's history
            assert len(recorded.env.env.traffic_data) == episode + 1
    finally:
        normal.close()
        recorded.close()

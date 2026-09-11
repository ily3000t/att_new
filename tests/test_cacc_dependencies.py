"""The native CACC path must not require optional traffic simulators."""

from pathlib import Path
import subprocess
import sys


def test_cacc_runs_without_sumo_imports():
    program = """
import importlib.abc
import sys

class RejectSumo(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'traci', 'sumolib', 'libsumo'}:
            raise ModuleNotFoundError('SUMO intentionally unavailable')

sys.meta_path.insert(0, RejectSumo())
import numpy as np
from amb.envs.network.network_env import NetworkEnv
for scenario in ('catchup', 'slowdown'):
    env = NetworkEnv({
        'scenario': scenario,
        'network_cfg': f'config_ma2c_nc_{scenario}.ini',
    })
    assert env.reset()[0].shape == (8, 5)
    transition = env.step(np.full((8, 1), 3))
    assert np.isfinite(transition[0]).all()
    assert np.isfinite(transition[2]).all()
    env.close()
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr

import json
from pathlib import Path

import pytest

from amb.utils import run_manifest


def test_manifest_captures_resolved_ini_git_and_checkpoint_lineage(tmp_path, monkeypatch):
    # Avoid capturing the developer's uncommitted files during this unit test.
    state = {"commit": "a" * 40, "branch": "test", "dirty": False, "status": []}
    monkeypatch.setattr(run_manifest, "source_state", lambda: state.copy())
    victim_run = tmp_path / "victim"
    model_dir = victim_run / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "actor.pth").write_bytes(b"checkpoint fixture")
    (victim_run / "manifest.json").write_text(json.dumps({
        "run_id": "victim-run", "source": {"commit": "b" * 40},
        "config_sha256": "c" * 64, "seeds": {"process_seed": 3},
    }))
    directory = tmp_path / "evaluation"
    directory.mkdir()
    config = {
        "algo_args": {
            "train": {"seed": 7, "n_rollout_threads": 2, "n_eval_rollout_threads": 2, "cuda": False},
            "victim": {"model_dir": str(model_dir)},
        },
        "env_args": {"network_cfg": "config_ma2c_nc_catchup.ini", "override_args": {"speed_target": 14}},
    }
    (directory / "config.json").write_text(json.dumps(config))
    data = run_manifest.write_run_manifest(config, directory)
    assert data["source"]["commit"] == "a" * 40
    assert data["seeds"]["train_worker_initial_seeds"] == [7, 1007]
    assert data["seeds"]["eval_worker_initial_seeds"] == [350000, 360000]
    assert "speed_target = 14" in (directory / "environment.ini").read_text()
    checkpoint = data["input_checkpoints"]["victim"]
    assert checkpoint["training_run"]["commit"] == "b" * 40
    old_hash = checkpoint["files"][0]["sha256"]
    (model_dir / "actor.pth").write_bytes(b"different fixture")
    assert run_manifest.checkpoint_inventory(model_dir)["files"][0]["sha256"] != old_hash
    run_manifest.finish_run_manifest(directory)
    finished = json.loads((directory / "manifest.json").read_text())
    assert finished["status"] == "completed"
    run_manifest.finish_run_manifest(directory, {"type": "TestError"})
    assert json.loads((directory / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("state", [{"commit": None, "dirty": None}, {"commit": "a" * 40, "dirty": True}])
def test_formal_runs_require_known_clean_commit(monkeypatch, state):
    monkeypatch.setattr(run_manifest, "source_state", lambda: state)
    with pytest.raises(RuntimeError, match="clean source tree"):
        run_manifest.require_clean_source()

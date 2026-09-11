"""Provenance for CACC runs and the checkpoints they consume/produce."""

import configparser
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args):
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args], stderr=subprocess.PIPE,
    )


def source_state():
    try:
        status = git_output("status", "--porcelain=v1", "-z")
        return {
            "commit": git_output("rev-parse", "HEAD").decode().strip(),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
            "dirty": bool(status),
            "status": status.decode("utf-8", errors="replace").split("\0")[:-1],
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "branch": None, "dirty": None, "error": type(exc).__name__}


def require_clean_source():
    state = source_state()
    if not state["commit"] or state["dirty"] is not False:
        raise RuntimeError("A committed, clean source tree is required; commit changes before this run.")
    return state


def checkpoint_inventory(model_dir, include_lineage=True):
    if not model_dir:
        return None
    directory = Path(model_dir).resolve()
    result = {"directory": str(directory), "exists": directory.is_dir(), "files": []}
    if directory.is_dir():
        for path in sorted(directory.rglob("*.pth")):
            result["files"].append({"path": path.relative_to(directory).as_posix(), "sha256": sha256_file(path)})
    # Standard models/ and slice/<step>/ both belong to the same training run.
    parent_manifest = next((parent / "manifest.json" for parent in (directory.parent, directory.parent.parent)
                            if (parent / "manifest.json").is_file()), None)
    if include_lineage and parent_manifest is not None:
        parent = json.loads(parent_manifest.read_text(encoding="utf-8"))
        result["training_run"] = {
            "manifest_sha256": sha256_file(parent_manifest),
            "run_id": parent.get("run_id"),
            "commit": parent.get("source", {}).get("commit"),
            "config_sha256": parent.get("config_sha256"),
            "seed": parent.get("seeds", {}).get("process_seed"),
            "checkpoint_subdirectory": directory.relative_to(parent_manifest.parent).as_posix(),
        }
    return result


def write_run_manifest(config, run_dir):
    import torch

    directory = Path(run_dir).resolve()
    source = source_state()
    if source["dirty"]:
        (directory / "source_dirty.patch").write_bytes(git_output("diff", "--binary", "HEAD"))
        source["untracked_snapshots"] = []
        for name in git_output("ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0"):
            if not name:
                continue
            path = (ROOT / name).resolve()
            if path.is_relative_to(directory) or not path.is_file():
                continue  # never recursively copy this run's own output
            relative = path.relative_to(ROOT)
            target = directory / "source_untracked" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            source["untracked_snapshots"].append({"path": relative.as_posix(), "sha256": sha256_file(target)})
    train = config["algo_args"]["train"]
    victim = config["algo_args"].get("victim", {})
    seed = int(train["seed"])
    env_args = config["env_args"]
    ini_path = ROOT / "amb/envs/network/config" / env_args["network_cfg"]
    ini = configparser.ConfigParser()
    with ini_path.open(encoding="utf-8") as stream:
        ini.read_file(stream)
    for key, value in (env_args.get("override_args") or {}).items():
        if key in ini["ENV_CONFIG"]:
            ini["ENV_CONFIG"][key] = str(value)
    with (directory / "environment.ini").open("w", encoding="utf-8") as stream:
        ini.write(stream)
    data = {
        "schema_version": 1, "run_id": uuid.uuid4().hex,
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "classification": "committed" if source["dirty"] is False else "exploratory",
        "purpose": config["algo_args"].get("extra", {}).get("purpose", "unspecified"),
        "source": source,
        "command": [sys.executable, *sys.argv], "working_directory": str(Path.cwd()),
        "config_sha256": sha256_file(directory / "config.json"),
        "environment_ini_sha256": sha256_file(directory / "environment.ini"),
        "seeds": {
            "process_seed": seed,
            "train_worker_initial_seeds": [seed + rank * 1000 for rank in range(train["n_rollout_threads"])],
            "eval_worker_initial_seeds": [seed * 50000 + rank * 10000 for rank in range(train["n_eval_rollout_threads"])],
            "episode_rule": "first reset uses initial seed, subsequent resets increment it by one",
            "python_numpy_torch": "process_seed; CACC uses an independent RandomState",
            "attack_rng": "process Python/NumPy/Torch RNGs; not an independent attack seed",
            "eval_seed_override": config["algo_args"].get("extra", {}).get("eval_seed_override"),
        },
        "runtime": {
            "python": sys.version, "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
            "device": "cuda:0" if train["cuda"] and torch.cuda.is_available() else "cpu",
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "torch_threads": torch.get_num_threads(),
            "packages": dict(sorted((dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions() if dist.metadata["Name"])),
        },
        "input_checkpoints": {
            "victim": checkpoint_inventory(victim.get("model_dir")),
            "train": checkpoint_inventory(train.get("model_dir")),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return data


def finish_run_manifest(run_dir, error=None):
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update({
        "status": "failed" if error else "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "output_checkpoints": checkpoint_inventory(Path(run_dir) / "models", include_lineage=False),
    })
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")

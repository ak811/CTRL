"""Filesystem helpers: run directories, CSV/JSON/YAML writers and provenance metadata."""

from __future__ import annotations

import csv
import datetime as _dt
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml

from common.config import config_to_dict

REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(root: str | Path, *parts: str, run_name: str | None = None) -> Path:
    """Create ``root/parts.../<run_name or timestamp>`` without clobbering existing runs."""
    base = Path(root).joinpath(*parts)
    name = run_name or timestamp()
    candidate, suffix = base / name, 1
    while candidate.exists():
        candidate = base / f"{name}_{suffix}"
        suffix += 1
    return ensure_dir(candidate)


def write_csv(path: str | Path, rows: Iterable[Sequence[Any]], header: Sequence[str] | None = None) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


def write_yaml(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        )
        if out.returncode != 0:
            return None
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def collect_metadata() -> dict[str, Any]:
    """Capture enough provenance to reproduce a run."""
    meta: dict[str, Any] = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_revision(),
        "packages": {},
    }
    for pkg in ("numpy", "torch", "gymnasium", "stable_baselines3", "ale_py", "cv2"):
        try:
            module = __import__(pkg)
            meta["packages"][pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            continue
    try:
        import torch

        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:  # pragma: no cover
        pass
    return meta


def save_run_metadata(run_dir: str | Path, cfg: Any) -> None:
    """Write ``config.yaml`` (resolved config) and ``metadata.json`` (provenance)."""
    run_dir = Path(run_dir)
    write_yaml(run_dir / "config.yaml", config_to_dict(cfg))
    write_json(run_dir / "metadata.json", collect_metadata())

"""Training configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic.v1 import ValidationError

from app.ml.common import paths
from app.ml.training.schemas import TrainingConfig


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve(strict=False)
        repo_candidate = (paths.get_repository_root() / candidate).resolve(strict=False)
        if cwd_candidate.exists() and paths.is_path_inside(paths.get_repository_root(), cwd_candidate):
            candidate = cwd_candidate
        elif repo_candidate.exists() or not str(candidate).replace("\\", "/").startswith("../"):
            candidate = repo_candidate
        else:
            candidate = cwd_candidate
    return candidate.resolve(strict=False)


def load_training_config(path: str | Path) -> TrainingConfig:
    resolved = _resolve_project_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    try:
        return TrainingConfig.parse_obj(payload)
    except ValidationError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse training config {resolved}: {exc}") from exc


def training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    return config.to_safe_dict()

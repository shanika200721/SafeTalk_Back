"""Reproducibility controls and portable environment capture."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path
from typing import Any, Iterable, Mapping
import uuid

import numpy as np

from app.ml.common import hashing, paths
from app.ml.training.schemas import TrainingConfig


def set_global_seed(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)


def hash_training_config(config: TrainingConfig | Mapping[str, Any]) -> str:
    payload = config.to_safe_dict() if hasattr(config, "to_safe_dict") else dict(config)
    return hashing.hash_json_data(payload)


def deterministic_run_id(config: TrainingConfig | Mapping[str, Any], *, salt: str = "") -> str:
    digest = hashing.hash_text(f"{hash_training_config(config)}:{salt}")[:32]
    return f"run-{uuid.UUID(digest)}"


def capture_python_version() -> dict[str, str]:
    return {"python": platform.python_version(), "implementation": platform.python_implementation()}


def _module_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def capture_library_versions() -> dict[str, str | None]:
    return {
        "numpy": _module_version("numpy"),
        "pandas": _module_version("pandas"),
        "scikit-learn": _module_version("sklearn"),
        "scipy": _module_version("scipy"),
    }


def capture_platform_summary() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    }


def capture_environment_versions() -> dict[str, Any]:
    return {
        "python": capture_python_version(),
        "libraries": capture_library_versions(),
        "platform": capture_platform_summary(),
    }


def validate_deterministic_estimator_settings(estimator: Any) -> list[str]:
    warnings: list[str] = []
    params = estimator.get_params(deep=True) if hasattr(estimator, "get_params") else {}
    if "random_state" in params and params.get("random_state") is None:
        warnings.append("estimator exposes random_state but it is unset")
    if "shuffle" in params and params.get("shuffle") is True and params.get("random_state") is None:
        warnings.append("estimator shuffles without a fixed random_state")
    if "n_jobs" in params and params.get("n_jobs") not in (None, 1):
        warnings.append("parallel estimator execution may reduce determinism")
    return warnings


def _assert_portable_report(report: Mapping[str, Any]) -> None:
    text = json.dumps(report, sort_keys=True, default=str)
    forbidden_tokens = ["PASSWORD", "SECRET", "TOKEN", "DATABASE_URL", "postgresql://", "\\Users\\", "/home/"]
    if any(token.lower() in text.lower() for token in forbidden_tokens):
        raise ValueError("reproducibility report contains secret-like or local path content")


def save_reproducibility_report(report: Mapping[str, Any], output_path: str | Path, *, overwrite: bool = False) -> Path:
    _assert_portable_report(report)
    candidate = Path(output_path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    resolved = candidate.resolve(strict=False)
    if not paths.is_path_inside(paths.get_generated_root(), resolved) and not paths.is_path_inside(paths.get_model_root(), resolved):
        raise ValueError("reproducibility reports must be saved under generated/ or MODEL_ROOT")
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite reproducibility report: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_name(f".{resolved.name}.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(resolved)
    return resolved

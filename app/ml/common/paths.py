"""Central path policy for SafeTalk ML assets.

Raw datasets are read-only inputs. Generated outputs and model artifacts must
stay outside ``Final Dataset/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

try:
    from app.core.config import settings
except Exception:  # pragma: no cover - defensive fallback for isolated imports
    settings = None


PathLike = Union[str, os.PathLike[str], Path]

_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = _THIS_FILE.parents[3]
_REPOSITORY_ROOT = _THIS_FILE.parents[4]

_RAW_DATASET_ROOT = _REPOSITORY_ROOT / "Final Dataset"
_ML_RESEARCH_ROOT = _REPOSITORY_ROOT / "ml-research"
_GENERATED_ROOT = _REPOSITORY_ROOT / "generated"
_GENERATED_PREPROCESSING_ROOT = _GENERATED_ROOT / "preprocessing"
_GENERATED_MANIFESTS_ROOT = _GENERATED_ROOT / "manifests"
_GENERATED_REPORTS_ROOT = _GENERATED_ROOT / "reports"
_GENERATED_AUDITS_ROOT = _GENERATED_ROOT / "audits"
_GENERATED_TEMPORARY_ROOT = _GENERATED_ROOT / "temporary"
_CANONICAL_MODEL_ROOT = _REPOSITORY_ROOT / "ml_models"

_MODEL_ROOT_ALIASES = {
    ".",
    "ml_models",
    "./ml_models",
    ".\\ml_models",
    "../ml_models",
    "..\\ml_models",
}


def _resolve_existing_policy_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def get_repository_root() -> Path:
    """Return the repository root, independent of the current working directory."""
    return _REPOSITORY_ROOT


def get_backend_root() -> Path:
    """Return the backend package root."""
    return _BACKEND_ROOT


def get_raw_dataset_root() -> Path:
    """Return the immutable raw dataset root without creating it."""
    return _RAW_DATASET_ROOT


def get_ml_research_root() -> Path:
    """Return the versioned ML research workspace root."""
    return _ML_RESEARCH_ROOT


def get_generated_root() -> Path:
    """Return the root for reproducible generated outputs."""
    return _GENERATED_ROOT


def get_generated_preprocessing_root() -> Path:
    return _GENERATED_PREPROCESSING_ROOT


def get_generated_manifests_root() -> Path:
    return _GENERATED_MANIFESTS_ROOT


def get_generated_reports_root() -> Path:
    return _GENERATED_REPORTS_ROOT


def _settings_model_root() -> str:
    if settings is not None and getattr(settings, "MODEL_ROOT", None):
        return str(settings.MODEL_ROOT)
    return "../ml_models"


def _normalize_model_root_value(model_root_value: PathLike) -> str:
    return str(model_root_value).strip().replace("\\", "/").rstrip("/")


def resolve_model_root(model_root_value: Optional[PathLike] = None) -> Path:
    """Resolve a MODEL_ROOT value to an absolute path.

    The canonical local model root is the repository-level ``ml_models/``.
    Common relative values such as ``../ml_models`` and ``./ml_models`` both
    resolve there so app behavior does not depend on the launch directory.
    Other relative overrides are resolved against the backend root.
    """
    raw_value = _settings_model_root() if model_root_value is None else model_root_value
    value = Path(os.path.expandvars(str(raw_value))).expanduser()

    if value.is_absolute():
        return value.resolve(strict=False)

    normalized = _normalize_model_root_value(value)
    if normalized in _MODEL_ROOT_ALIASES or value.name == "ml_models":
        return _CANONICAL_MODEL_ROOT.resolve(strict=False)

    return (_BACKEND_ROOT / value).resolve(strict=False)


def get_model_root() -> Path:
    """Return the resolved model artifact root."""
    return resolve_model_root()


def is_path_inside(parent: PathLike, child: PathLike) -> bool:
    """Return True when child is parent or inside parent after resolution."""
    parent_path = _resolve_existing_policy_path(Path(parent))
    child_path = _resolve_existing_policy_path(Path(child))
    try:
        child_path.relative_to(parent_path)
        return True
    except ValueError:
        return False


def _project_path(path: PathLike) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = _REPOSITORY_ROOT / candidate
    return candidate.resolve(strict=False)


def assert_not_raw_dataset_path(path: PathLike) -> Path:
    """Raise if path points to or inside the immutable raw dataset root."""
    candidate = _project_path(path)
    raw_root = _RAW_DATASET_ROOT.resolve(strict=False)
    if is_path_inside(raw_root, candidate):
        raise ValueError(f"Refusing to write inside raw dataset root: {candidate}")
    return candidate


def _assert_no_traversal(relative_path: Path) -> None:
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative path, got: {relative_path}")
    if any(part == ".." for part in relative_path.parts):
        raise ValueError(f"Path traversal is not allowed: {relative_path}")


def _assert_approved_output_path(path: PathLike) -> Path:
    candidate = assert_not_raw_dataset_path(path)
    approved_roots = (
        _GENERATED_ROOT.resolve(strict=False),
        _ML_RESEARCH_ROOT.resolve(strict=False),
        _CANONICAL_MODEL_ROOT.resolve(strict=False),
    )
    if not any(is_path_inside(root, candidate) for root in approved_roots):
        raise ValueError(f"Path is outside approved generated/model roots: {candidate}")
    return candidate


def resolve_generated_path(relative_path: PathLike) -> Path:
    """Resolve a relative generated output path under ``generated/`` safely."""
    relative = Path(relative_path)
    _assert_no_traversal(relative)
    target = (_GENERATED_ROOT / relative).resolve(strict=False)
    return _assert_approved_output_path(target)


def ensure_generated_directories() -> tuple[Path, ...]:
    """Create only approved Phase 2 generated and research directories."""
    directories = (
        _ML_RESEARCH_ROOT,
        _ML_RESEARCH_ROOT / "configs",
        _ML_RESEARCH_ROOT / "manifests",
        _ML_RESEARCH_ROOT / "preprocessing",
        _ML_RESEARCH_ROOT / "preprocessing" / "common",
        _ML_RESEARCH_ROOT / "preprocessing" / "dass21",
        _ML_RESEARCH_ROOT / "preprocessing" / "profile",
        _ML_RESEARCH_ROOT / "preprocessing" / "mood",
        _ML_RESEARCH_ROOT / "preprocessing" / "text",
        _ML_RESEARCH_ROOT / "preprocessing" / "speech",
        _ML_RESEARCH_ROOT / "preprocessing" / "face",
        _ML_RESEARCH_ROOT / "preprocessing" / "behavioral",
        _ML_RESEARCH_ROOT / "reports",
        _ML_RESEARCH_ROOT / "tests",
        _GENERATED_ROOT,
        _GENERATED_PREPROCESSING_ROOT,
        _GENERATED_MANIFESTS_ROOT,
        _GENERATED_REPORTS_ROOT,
        _GENERATED_AUDITS_ROOT,
        _GENERATED_TEMPORARY_ROOT,
        _CANONICAL_MODEL_ROOT,
    )

    created_or_existing = []
    for directory in directories:
        safe_directory = _assert_approved_output_path(directory)
        safe_directory.mkdir(parents=True, exist_ok=True)
        created_or_existing.append(safe_directory)
    return tuple(created_or_existing)


def ensure_model_directory(relative_path: Optional[PathLike] = None) -> Path:
    """Create the model root or a child directory under it."""
    model_root = get_model_root()
    if relative_path is None:
        target = model_root
    else:
        relative = Path(relative_path)
        _assert_no_traversal(relative)
        target = (model_root / relative).resolve(strict=False)

    target = assert_not_raw_dataset_path(target)
    if not is_path_inside(model_root, target):
        raise ValueError(f"Model directory must remain inside MODEL_ROOT: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target

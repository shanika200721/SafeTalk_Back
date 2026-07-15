"""Read-only hashing helpers for ML configs, manifests, and datasets."""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from app.ml.common import paths


DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024
SUPPORTED_DATASET_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".xlsx",
    ".txt",
    ".wav",
    ".mp3",
    ".flac",
    ".jpg",
    ".jpeg",
    ".png",
}
DEFAULT_IGNORED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
    ".ipynb_checkpoints",
    "__pycache__",
    ".pytest_cache",
}
DATASET_FINGERPRINT_VERSION = "1.0.0"

_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_text(text: str, encoding: str = "utf-8") -> str:
    """Return the SHA-256 digest for text encoded with ``encoding``."""
    return hash_bytes(text.encode(encoding))


def sha256_text(value: str) -> str:
    """Backward-compatible alias for hashing UTF-8 text."""
    return hash_text(value)


def hash_json_data(data: Any) -> str:
    """Return a deterministic SHA-256 digest for JSON-like data."""
    return hash_text(_stable_json(data))


def sha256_json(value: Mapping[str, Any]) -> str:
    """Backward-compatible alias for deterministic JSON metadata hashing."""
    return hash_json_data(value)


def _as_resolved_path(path: str | os.PathLike[str] | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.expanduser().resolve(strict=False)


def _assert_approved_read_path(path: Path, *, allow_outside_project: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if allow_outside_project:
        return resolved
    if not paths.is_path_inside(paths.get_repository_root(), resolved):
        raise ValueError(f"Path is outside approved project locations: {resolved}")
    return resolved


def _is_absolute_report_path(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return normalized.startswith("/") or normalized.startswith("//") or bool(_WINDOWS_DRIVE_RE.match(value))


def _assert_safe_relative_report_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise ValueError("relative path cannot be blank")
    if _is_absolute_report_path(normalized):
        raise ValueError(f"relative path must not be absolute: {value}")
    parts = [part for part in normalized.split("/") if part]
    if any(part == ".." for part in parts):
        raise ValueError(f"relative path must not contain traversal: {value}")
    return "/".join(parts)


def normalize_relative_path(path: str | os.PathLike[str] | Path, root: str | os.PathLike[str] | Path) -> str:
    """Return a POSIX-style relative path from ``root`` to ``path``.

    Both inputs are resolved first so current working directory changes do not
    alter report paths.
    """
    root_path = _as_resolved_path(root)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root_path / candidate
    resolved = candidate.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"Path is outside requested root: {resolved}") from exc
    return _assert_safe_relative_report_path(relative.as_posix())


def _default_report_root(source_path: Path, *, allow_outside_project: bool) -> Path:
    repository_root = paths.get_repository_root().resolve(strict=False)
    if paths.is_path_inside(repository_root, source_path):
        return repository_root
    if allow_outside_project:
        return source_path if source_path.is_dir() else source_path.parent
    return repository_root


def sha256_file(
    path: str | os.PathLike[str] | Path,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    *,
    allow_outside_project: bool = False,
) -> str:
    """Return a SHA-256 digest for a file without loading it all at once."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    resolved = _assert_approved_read_path(_as_resolved_path(path), allow_outside_project=allow_outside_project)
    if not resolved.exists():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Expected a file: {resolved}")

    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(
    path: str | os.PathLike[str] | Path,
    expected_hash: str,
    *,
    allow_outside_project: bool = False,
) -> bool:
    """Return True when the current file SHA-256 matches ``expected_hash``."""
    return sha256_file(path, allow_outside_project=allow_outside_project) == expected_hash.lower()


def _normalized_extensions(extensions: Optional[Iterable[str]]) -> Optional[set[str]]:
    if extensions is None:
        return None
    normalized = set()
    for extension in extensions:
        value = str(extension).strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        normalized.add(value)
    return normalized


def _ignored_name_reason(name: str, ignored_names: set[str]) -> Optional[str]:
    if name in ignored_names:
        return "ignored_cache_or_temporary_name"
    if name.startswith("~$"):
        return "ignored_temporary_office_file"
    if name.startswith(".") and (name.endswith(".tmp") or name.endswith(".swp") or name.endswith(".part")):
        return "ignored_hidden_temporary_file"
    return None


def _media_type(path: Path) -> Optional[str]:
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed


def _file_entry(
    path: Path,
    report_root: Path,
    *,
    include_modified_time: bool,
    allow_outside_project: bool,
) -> dict[str, Any]:
    stat = path.stat()
    payload: dict[str, Any] = {
        "relative_path": normalize_relative_path(path, report_root),
        "sha256": sha256_file(path, allow_outside_project=allow_outside_project),
        "size_bytes": stat.st_size,
        "extension": path.suffix.lower().lstrip("."),
    }
    if include_modified_time:
        payload["modified_time_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    media_type = _media_type(path)
    if media_type:
        payload["media_type"] = media_type
    return payload


def create_file_fingerprint(
    path: str | os.PathLike[str] | Path,
    *,
    root: str | os.PathLike[str] | Path | None = None,
    include_modified_time: bool = False,
    allow_outside_project: bool = False,
) -> dict[str, Any]:
    """Create a read-only fingerprint entry for one file."""
    resolved = _assert_approved_read_path(_as_resolved_path(path), allow_outside_project=allow_outside_project)
    if not resolved.exists():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Expected a file: {resolved}")
    report_root = _as_resolved_path(root) if root is not None else _default_report_root(resolved, allow_outside_project=allow_outside_project)
    return _file_entry(
        resolved,
        report_root,
        include_modified_time=include_modified_time,
        allow_outside_project=allow_outside_project,
    )


def _combined_directory_hash(files: list[dict[str, Any]]) -> str:
    entries = [
        {
            "relative_path": file_entry["relative_path"],
            "sha256": file_entry["sha256"],
            "size_bytes": file_entry["size_bytes"],
        }
        for file_entry in sorted(files, key=lambda item: item["relative_path"])
    ]
    return hash_json_data(entries)


def create_directory_fingerprint(
    path: str | os.PathLike[str] | Path,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    ignored_names: Optional[Iterable[str]] = None,
    root: str | os.PathLike[str] | Path | None = None,
    include_modified_time: bool = False,
    allow_empty: bool = False,
    allow_outside_project: bool = False,
) -> dict[str, Any]:
    """Create a deterministic read-only fingerprint for a directory tree."""
    resolved = _assert_approved_read_path(_as_resolved_path(path), allow_outside_project=allow_outside_project)
    if not resolved.exists():
        raise FileNotFoundError(f"Directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Expected a directory: {resolved}")

    report_root = _as_resolved_path(root) if root is not None else _default_report_root(resolved, allow_outside_project=allow_outside_project)
    extension_filter = _normalized_extensions(allowed_extensions) or SUPPORTED_DATASET_EXTENSIONS
    ignored = set(DEFAULT_IGNORED_NAMES)
    if ignored_names:
        ignored.update(str(name) for name in ignored_names)

    files: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    root_real = resolved.resolve(strict=True)

    for current_root, dirnames, filenames in os.walk(resolved, topdown=True, followlinks=False):
        current = Path(current_root)

        retained_dirs = []
        for dirname in sorted(dirnames):
            child = current / dirname
            relative = normalize_relative_path(child, report_root)
            ignored_reason = _ignored_name_reason(dirname, ignored)
            if ignored_reason:
                skipped.append({"relative_path": relative, "reason": ignored_reason})
                continue
            if child.is_symlink():
                target = child.resolve(strict=False)
                if not paths.is_path_inside(root_real, target):
                    raise ValueError(f"Symbolic link escapes dataset root: {child} -> {target}")
                skipped.append({"relative_path": relative, "reason": "symlink_directory_not_followed"})
                continue
            retained_dirs.append(dirname)
        dirnames[:] = retained_dirs

        for filename in sorted(filenames):
            file_path = current / filename
            relative = normalize_relative_path(file_path, report_root)
            ignored_reason = _ignored_name_reason(filename, ignored)
            if ignored_reason:
                skipped.append({"relative_path": relative, "reason": ignored_reason})
                continue

            if file_path.is_symlink():
                target = file_path.resolve(strict=False)
                if not paths.is_path_inside(root_real, target):
                    raise ValueError(f"Symbolic link escapes dataset root: {file_path} -> {target}")

            extension = file_path.suffix.lower()
            if extension not in extension_filter:
                skipped.append({"relative_path": relative, "reason": f"unsupported_extension:{extension or '<none>'}"})
                continue

            files.append(
                _file_entry(
                    file_path,
                    report_root,
                    include_modified_time=include_modified_time,
                    allow_outside_project=allow_outside_project,
                )
            )

    files.sort(key=lambda item: item["relative_path"])
    skipped.sort(key=lambda item: (item["relative_path"], item["reason"]))
    if not files and not allow_empty:
        raise ValueError(f"Directory contains no fingerprintable files: {resolved}")

    return {
        "source_relative_path": normalize_relative_path(resolved, report_root),
        "source_type": "directory",
        "file_count": len(files),
        "total_bytes": sum(file_entry["size_bytes"] for file_entry in files),
        "combined_sha256": _combined_directory_hash(files),
        "files": files,
        "skipped_files": skipped,
        "skipped_file_count": len(skipped),
    }


def hash_directory(
    path: str | os.PathLike[str] | Path,
    *,
    allowed_extensions: Optional[Iterable[str]] = None,
    ignored_names: Optional[Iterable[str]] = None,
    allow_empty: bool = False,
    allow_outside_project: bool = False,
) -> str:
    """Return the deterministic combined SHA-256 for a directory tree."""
    fingerprint = create_directory_fingerprint(
        path,
        allowed_extensions=allowed_extensions,
        ignored_names=ignored_names,
        allow_empty=allow_empty,
        allow_outside_project=allow_outside_project,
    )
    return fingerprint["combined_sha256"]

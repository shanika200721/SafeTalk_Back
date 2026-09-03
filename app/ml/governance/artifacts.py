"""Read-only artifact discovery and integrity validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from app.ml.common import paths
from app.ml.governance.constants import EVALUATION_ONLY_ROOT_MARKERS, REAL_BASELINE_MODALITIES, SYNTHETIC_MODALITIES
from app.ml.governance.schemas import ArtifactIntegrityStatus


@dataclass(frozen=True)
class DiscoveredModelRun:
    modality: str
    model_name: str
    model_version: str
    run_id: str
    run_path: str
    manifest_path: str
    selected_candidate: bool = False
    synthetic: bool = False
    evaluation_only: bool = False
    unexpected: bool = False
    manifest: Optional[Dict[str, Any]] = None
    status: ArtifactIntegrityStatus = ArtifactIntegrityStatus.not_applicable
    findings: List[str] = field(default_factory=list)


def repo_relative(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    try:
        return candidate.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return candidate.resolve(strict=False).as_posix()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else paths.get_repository_root() / candidate


def _read_json(path: str | Path) -> Dict[str, Any]:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with resolve_repo_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact_manifest(path: str | Path) -> Dict[str, Any]:
    return _read_json(path)


def _is_evaluation_only(path: Path) -> bool:
    normalized = path.as_posix().lower()
    return any(marker in normalized for marker in EVALUATION_ONLY_ROOT_MARKERS)


def _selected_run_paths(reports_root: str | Path | None = None) -> Dict[str, str]:
    root = resolve_repo_path(reports_root or paths.get_generated_reports_root())
    selections: Dict[str, str] = {}
    summary_files = {
        "profile": root / "profile_baseline" / "v1" / "profile_baseline_summary.json",
        "text": root / "text_baseline" / "v1" / "text_baseline_summary.json",
        "speech": root / "speech_baseline" / "v1" / "speech_baseline_summary.json",
        "face": root / "face_baseline" / "v1" / "face_baseline_summary.json",
    }
    for modality, summary_path in summary_files.items():
        summary: Dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = _read_json(summary_path)
                artifact_path = summary.get("artifact_path")
            except Exception:
                artifact_path = None
            if artifact_path:
                selections[modality] = repo_relative(artifact_path)
            elif summary.get("model_name") and summary.get("model_version") and summary.get("run_id"):
                selections[modality] = repo_relative(
                    paths.get_model_root() / modality / summary["model_name"] / summary["model_version"] / summary["run_id"]
                )
    return selections


def discover_model_runs(
    model_root: str | Path | None = None,
    reports_root: str | Path | None = None,
    modalities: Optional[Iterable[str]] = None,
    include_evaluation_only: bool = True,
) -> List[DiscoveredModelRun]:
    root = resolve_repo_path(model_root or paths.get_model_root())
    requested = {item.lower() for item in modalities or REAL_BASELINE_MODALITIES}
    selected_paths = _selected_run_paths(reports_root)
    runs: List[DiscoveredModelRun] = []
    for manifest_path in sorted(root.rglob("artifact_manifest.json")):
        rel_manifest = repo_relative(manifest_path)
        rel_run = repo_relative(manifest_path.parent)
        synthetic = any(part.lower() in SYNTHETIC_MODALITIES for part in manifest_path.parts)
        evaluation_only = _is_evaluation_only(manifest_path)
        if evaluation_only and not include_evaluation_only:
            continue
        try:
            manifest = load_artifact_manifest(manifest_path)
            modality = str(manifest.get("modality") or manifest_path.relative_to(root).parts[0]).lower()
            model_name = str(manifest.get("model_name") or manifest_path.parent.parent.parent.name)
            model_version = str(manifest.get("model_version") or manifest_path.parent.parent.name)
            run_id = str(manifest.get("run_id") or manifest_path.parent.name)
            status = verify_artifact_manifest(manifest, manifest_path.parent)
            findings: List[str] = []
        except Exception as exc:
            modality = manifest_path.relative_to(root).parts[0].lower()
            model_name = manifest_path.parent.parent.parent.name if len(manifest_path.parts) >= 4 else "unknown"
            model_version = manifest_path.parent.parent.name if len(manifest_path.parts) >= 3 else "unknown"
            run_id = manifest_path.parent.name
            manifest = None
            status = ArtifactIntegrityStatus.malformed
            findings = [f"malformed_manifest: {exc}"]
        if modality not in requested and not evaluation_only:
            continue
        selected = selected_paths.get(modality) == rel_run
        unexpected = modality not in REAL_BASELINE_MODALITIES or synthetic
        runs.append(
            DiscoveredModelRun(
                modality=modality,
                model_name=model_name,
                model_version=model_version,
                run_id=run_id,
                run_path=rel_run,
                manifest_path=rel_manifest,
                selected_candidate=selected,
                synthetic=synthetic,
                evaluation_only=evaluation_only,
                unexpected=unexpected,
                manifest=manifest,
                status=status,
                findings=findings,
            )
        )
    return runs


def verify_artifact_manifest(manifest: Mapping[str, Any], run_dir: str | Path | None = None) -> ArtifactIntegrityStatus:
    required = {"manifest_version", "model_name", "model_version", "run_id", "files", "file_hashes"}
    if not required.issubset(set(manifest)):
        if manifest.get("run_id") and isinstance(manifest.get("files"), list) and isinstance(manifest.get("file_hashes"), dict):
            return verify_artifact_hashes(manifest)
        return ArtifactIntegrityStatus.malformed
    if not isinstance(manifest.get("files"), list) or not isinstance(manifest.get("file_hashes"), dict):
        return ArtifactIntegrityStatus.malformed
    if not manifest["files"]:
        return ArtifactIntegrityStatus.missing
    return verify_artifact_hashes(manifest)


def verify_artifact_hashes(manifest: Mapping[str, Any]) -> ArtifactIntegrityStatus:
    try:
        file_hashes = manifest["file_hashes"]
        files = manifest["files"]
    except KeyError:
        return ArtifactIntegrityStatus.malformed
    for file_entry in files:
        relative_path = file_entry.get("path") if isinstance(file_entry, Mapping) else file_entry
        if not relative_path:
            return ArtifactIntegrityStatus.malformed
        candidate = resolve_repo_path(relative_path)
        if not candidate.exists():
            return ArtifactIntegrityStatus.missing
        expected = file_hashes.get(relative_path)
        if not expected and isinstance(file_entry, Mapping):
            expected = file_entry.get("sha256")
        if not expected:
            return ArtifactIntegrityStatus.malformed
        if sha256_file(candidate) != expected:
            return ArtifactIntegrityStatus.hash_mismatch
    return ArtifactIntegrityStatus.verified


def verify_model_card_exists(run: DiscoveredModelRun | str | Path) -> ArtifactIntegrityStatus:
    run_path = resolve_repo_path(run.run_path if isinstance(run, DiscoveredModelRun) else run)
    return ArtifactIntegrityStatus.verified if (run_path / "model_card.md").exists() else ArtifactIntegrityStatus.missing


def verify_reproducibility_report(run: DiscoveredModelRun | str | Path) -> ArtifactIntegrityStatus:
    run_path = resolve_repo_path(run.run_path if isinstance(run, DiscoveredModelRun) else run)
    return ArtifactIntegrityStatus.verified if (run_path / "reproducibility_report.json").exists() else ArtifactIntegrityStatus.missing


def verify_split_reference(run: DiscoveredModelRun | str | Path) -> ArtifactIntegrityStatus:
    run_path = resolve_repo_path(run.run_path if isinstance(run, DiscoveredModelRun) else run)
    return ArtifactIntegrityStatus.verified if (run_path / "split_manifest_reference.json").exists() else ArtifactIntegrityStatus.missing


def verify_source_fingerprint_reference(manifest: Mapping[str, Any]) -> ArtifactIntegrityStatus:
    return ArtifactIntegrityStatus.verified if manifest.get("source_fingerprint") else ArtifactIntegrityStatus.missing


def verify_inactive_status(manifest: Mapping[str, Any]) -> ArtifactIntegrityStatus:
    return ArtifactIntegrityStatus.verified if manifest.get("active") is not True else ArtifactIntegrityStatus.malformed


def verify_registration_status(summary: Mapping[str, Any]) -> ArtifactIntegrityStatus:
    registered = summary.get("candidate_registration_occurred") or summary.get("registration_status") == "registered"
    return ArtifactIntegrityStatus.malformed if registered else ArtifactIntegrityStatus.verified


def detect_unexpected_model_artifacts(runs: Iterable[DiscoveredModelRun]) -> List[Dict[str, Any]]:
    findings = []
    for run in runs:
        if run.synthetic:
            findings.append({"run_path": run.run_path, "classification": "synthetic_excluded"})
        elif run.evaluation_only:
            findings.append({"run_path": run.run_path, "classification": "evaluation_only"})
        elif run.unexpected:
            findings.append({"run_path": run.run_path, "classification": "unexpected"})
    return findings


def build_artifact_integrity_summary(runs: Iterable[DiscoveredModelRun]) -> Dict[str, Any]:
    records = []
    counts: Dict[str, int] = {}
    selected_count = 0
    active_count = 0
    for run in runs:
        status = run.status.value if hasattr(run.status, "value") else str(run.status)
        counts[status] = counts.get(status, 0) + 1
        if run.selected_candidate:
            selected_count += 1
        if run.manifest and run.manifest.get("active") is True:
            active_count += 1
        records.append(
            {
                "modality": run.modality,
                "model_name": run.model_name,
                "model_version": run.model_version,
                "run_id": run.run_id,
                "run_path": run.run_path,
                "manifest_path": run.manifest_path,
                "selected_candidate": run.selected_candidate,
                "synthetic": run.synthetic,
                "evaluation_only": run.evaluation_only,
                "integrity_status": status,
                "findings": run.findings,
            }
        )
    return {
        "status_counts": counts,
        "selected_candidate_count": selected_count,
        "active_model_count": active_count,
        "records": records,
        "unexpected_artifacts": detect_unexpected_model_artifacts(runs),
        "synthetic_framework_artifacts_excluded": True,
        "speech_domain_evaluation_models_evaluation_only": True,
    }

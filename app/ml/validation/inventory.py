"""Read-only artifact inventory for Phase 2 validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from app.ml.common import paths
from app.ml.common.hashing import sha256_file
from app.ml.validation.constants import MODEL_ARTIFACT_EXTENSIONS
from app.ml.validation.schemas import ArtifactInventoryItem, repo_relative


EXPECTED_ARTIFACTS: dict[str, list[tuple[str, str]]] = {
    "dass21": [
        ("dataset config", "ml-research/configs/dass21.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/dass21.preprocessing.example.json"),
        ("field/label mapping", "ml-research/configs/dass21.item_mapping.v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/dass21-v1.json"),
        ("audit JSON", "generated/audits/dass21/v1/audit.json"),
        ("audit Markdown", "generated/audits/dass21/v1/audit.md"),
        ("preprocessing report", "generated/reports/dass21/scoring_validation_v1.json"),
    ],
    "profile": [
        ("dataset config", "ml-research/configs/profile.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/profile.preprocessing.example.json"),
        ("audit config", "ml-research/configs/profile.audit.example.json"),
        ("field/label mapping", "ml-research/configs/profile.field_mapping.v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/student-profile-v1.json"),
        ("audit JSON", "generated/audits/student-profile/v1/audit.json"),
        ("audit Markdown", "generated/audits/student-profile/v1/audit.md"),
        ("canonical data", "generated/preprocessing/profile/v1/canonical_profile.csv"),
        ("feature schema", "generated/preprocessing/profile/v1/profile_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/profile/v1/profile_preprocessing_report.json"),
        ("record manifest", "generated/preprocessing/profile/v1/profile_record_manifest.json"),
    ],
    "mood": [
        ("dataset config", "ml-research/configs/mood.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/mood.preprocessing.example.json"),
        ("source fingerprint", "generated/manifests/fingerprints/daily-mood-v1.json"),
        ("audit JSON", "generated/audits/mood/v1/audit.json"),
        ("feature schema", "generated/preprocessing/mood/v1/mood_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/mood/v1/mood_readiness_report.json"),
        ("feature table", "generated/preprocessing/mood/v1/generated-synthetic/mood_features.csv"),
        ("record manifest", "generated/preprocessing/mood/v1/generated-synthetic/mood_record_manifest.json"),
    ],
    "text": [
        ("dataset config", "ml-research/configs/text.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/text.preprocessing.example.json"),
        ("field/label mapping", "ml-research/configs/text.label_mapping.v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/mental-health-text-v1.json"),
        ("audit JSON", "generated/audits/text/v1/audit.json"),
        ("canonical data", "generated/preprocessing/text/v1/canonical_text.csv"),
        ("feature schema", "generated/preprocessing/text/v1/text_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/text/v1/text_preprocessing_report.json"),
        ("record manifest", "generated/preprocessing/text/v1/text_record_manifest.json"),
        ("duplicate/conflict manifest", "generated/preprocessing/text/v1/text_duplicate_manifest.json"),
        ("duplicate/conflict manifest", "generated/preprocessing/text/v1/text_conflict_quarantine.csv"),
    ],
    "speech": [
        ("dataset config", "ml-research/configs/speech.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/speech.preprocessing.example.json"),
        ("field/label mapping", "ml-research/configs/speech.label_mapping.v1.json"),
        ("field/label mapping", "ml-research/configs/speech.corpus_mapping.v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/speech/crema-v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/speech/ravdess-v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/speech/savee-v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/speech/tess-v1.json"),
        ("canonical data", "generated/preprocessing/speech/v1/speech_canonical_manifest.csv"),
        ("feature table", "generated/preprocessing/speech/v1/speech_features.csv"),
        ("feature schema", "generated/preprocessing/speech/v1/speech_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/speech/v1/speech_preprocessing_report.json"),
        ("record manifest", "generated/preprocessing/speech/v1/speech_record_manifest.json"),
        ("duplicate/conflict manifest", "generated/preprocessing/speech/v1/speech_duplicate_manifest.json"),
    ],
    "face": [
        ("dataset config", "ml-research/configs/face.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/face.preprocessing.example.json"),
        ("field/label mapping", "ml-research/configs/face.label_mapping.v1.json"),
        ("source fingerprint", "generated/manifests/fingerprints/face/facial-emotion-v1.json"),
        ("audit JSON", "generated/audits/face/v1/audit.json"),
        ("canonical data", "generated/preprocessing/face/v1/face_canonical_manifest.csv"),
        ("feature schema", "generated/preprocessing/face/v1/face_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/face/v1/face_preprocessing_report.json"),
        ("record manifest", "generated/preprocessing/face/v1/face_record_manifest.json"),
        ("duplicate/conflict manifest", "generated/preprocessing/face/v1/face_cross_label_conflicts.json"),
        ("duplicate/conflict manifest", "generated/preprocessing/face/v1/face_cross_split_overlap.json"),
    ],
    "behavioral": [
        ("dataset config", "ml-research/configs/behavioral.dataset.example.json"),
        ("preprocessing config", "ml-research/configs/behavioral.preprocessing.example.json"),
        ("field/label mapping", "ml-research/configs/behavioral.field_mapping.v1.json"),
        ("feature schema", "generated/preprocessing/behavioral/v1/behavioral_feature_schema.json"),
        ("preprocessing report", "generated/preprocessing/behavioral/v1/behavioral_readiness_report.json"),
        ("feature table", "generated/preprocessing/behavioral/v1/generated-synthetic/behavioral_features.csv"),
        ("record manifest", "generated/preprocessing/behavioral/v1/generated-synthetic/behavioral_record_manifest.json"),
    ],
}


def _resolve(relative_path: str | Path) -> Path:
    candidate = Path(relative_path)
    if not candidate.is_absolute():
        candidate = paths.get_repository_root() / candidate
    return candidate.resolve(strict=False)


def _extract_version(path: Path) -> Optional[str]:
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("version", "dataset_version", "preprocessing_version", "feature_schema_version", "fingerprint_version"):
        if payload.get(key):
            return str(payload[key])
    return None


def _item(modality: str, artifact_type: str, relative_path: str, *, expected: str = "expected") -> ArtifactInventoryItem:
    path = _resolve(relative_path)
    exists = path.exists()
    sha256 = None
    if exists and path.is_file() and not relative_path.replace("\\", "/").startswith("Final Dataset/"):
        sha256 = sha256_file(path)
    return ArtifactInventoryItem(
        modality=modality,
        artifact_type=artifact_type,
        relative_path=repo_relative(path),
        exists=exists,
        size=path.stat().st_size if exists and path.is_file() else 0,
        sha256=sha256,
        version=_extract_version(path),
        generated_status="generated" if relative_path.replace("\\", "/").startswith("generated/") else "source_or_config",
        source_generated_classification="generated" if relative_path.replace("\\", "/").startswith("generated/") else "source",
        expected_classification=expected,
    )


def create_phase2_artifact_inventory(
    *,
    modalities: Iterable[str] | None = None,
    generated_root: str | Path | None = None,
) -> list[ArtifactInventoryItem]:
    selected = list(modalities or EXPECTED_ARTIFACTS.keys())
    items: list[ArtifactInventoryItem] = []
    for modality in selected:
        for artifact_type, relative_path in EXPECTED_ARTIFACTS.get(modality, []):
            items.append(_item(modality, artifact_type, relative_path))

    search_root = _resolve(generated_root or paths.get_generated_root())
    if search_root.exists():
        for artifact in search_root.rglob("*"):
            if artifact.is_file() and artifact.suffix.lower() in MODEL_ARTIFACT_EXTENSIONS:
                items.append(
                    ArtifactInventoryItem(
                        modality="unknown",
                        artifact_type="model artifact",
                        relative_path=repo_relative(artifact),
                        exists=True,
                        size=artifact.stat().st_size,
                        sha256=sha256_file(artifact),
                        generated_status="generated",
                        source_generated_classification="generated",
                        expected_classification="unexpected",
                    )
                )
    return items

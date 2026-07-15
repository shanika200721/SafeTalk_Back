"""Canonical read-only preprocessing for speech emotion corpora."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import paths
from app.ml.common.schemas import DatasetFingerprint
from app.ml.preprocessing.speech.audio_io import audio_sha256, extract_audio_metadata as read_audio_metadata, load_wav_audio, write_generated_wav
from app.ml.preprocessing.speech.constants import (
    FEATURE_COLUMNS,
    RECORD_ID_PREFIX,
    SAFE_SPEAKER_KEY_PREFIX,
    SPEECH_CORPUS_MAPPING_VERSION,
    SPEECH_FEATURE_SCHEMA_VERSION,
    SPEECH_LABEL_MAPPING_VERSION,
    SPEECH_PREPROCESSING_VERSION,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from app.ml.preprocessing.speech.features import build_speech_feature_schema, extract_acoustic_features
from app.ml.preprocessing.speech.filename_parsers import parse_with_named_parser
from app.ml.preprocessing.speech.mapping import load_speech_corpus_mapping_config, load_speech_label_mapping_config, normalize_emotion_label
from app.ml.preprocessing.speech.reporting import create_speech_preprocessing_markdown
from app.ml.preprocessing.speech.schemas import (
    SpeechCanonicalRecord,
    SpeechCorpusConfig,
    SpeechCorpusMappingConfig,
    SpeechFeatureRecord,
    SpeechLabelMappingConfig,
    SpeechPreprocessingReport,
    SpeechSourceRecord,
)
from app.ml.preprocessing.speech.validation import detect_speaker_leakage_risk, validate_feature_values


def _resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    return (paths.get_repository_root() / candidate).resolve(strict=False)


def _relative_to_repo(path: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(paths.get_repository_root()).as_posix()
    except ValueError:
        return path.name


def discover_speech_files(corpus: SpeechCorpusConfig, *, max_files: int | None = None) -> list[Path]:
    root = _resolve_project_path(corpus.source_path)
    if not root.exists():
        raise FileNotFoundError(f"Speech corpus path does not exist: {corpus.source_path}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    return files[:max_files] if max_files is not None else files


def load_corpus_mapping(path: str | Path | None) -> SpeechCorpusMappingConfig:
    return load_speech_corpus_mapping_config(path)


def parse_speech_file(path: str | Path, corpus: SpeechCorpusConfig) -> SpeechSourceRecord:
    record = parse_with_named_parser(corpus.filename_parser, path)
    return record.copy(update={"source_file": _relative_to_repo(Path(path))})


def extract_audio_metadata(path: str | Path):
    return read_audio_metadata(path)


def generate_speech_record_id(corpus_name: str, source_relative_path: str, source_fingerprint: str) -> str:
    if len(source_fingerprint) < 12:
        raise ValueError("source_fingerprint must be available for deterministic speech record IDs")
    digest = hashlib.sha256(f"{RECORD_ID_PREFIX}:{corpus_name}:{source_relative_path}:{source_fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"{RECORD_ID_PREFIX}-{digest}"


def generate_safe_speaker_key(corpus_name: str, original_speaker_id: str, source_fingerprint: str) -> str:
    if not str(original_speaker_id).strip():
        raise ValueError("original_speaker_id is required")
    digest = hashlib.sha256(f"{SAFE_SPEAKER_KEY_PREFIX}:{corpus_name}:{original_speaker_id}:{source_fingerprint}".encode("utf-8")).hexdigest()[:16]
    return f"{SAFE_SPEAKER_KEY_PREFIX}-{digest}"


def normalize_emotion_label_for_record(record: SpeechSourceRecord, mapping_config: SpeechLabelMappingConfig) -> str:
    return normalize_emotion_label(record.corpus_name, record.original_emotion_label, mapping_config)


def extract_speech_features(path: str | Path) -> tuple[dict[str, float], list[str]]:
    return extract_acoustic_features(path)


def _duration_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    series = pd.Series(values, dtype="float64")
    return {
        "min": round(float(series.min()), 6),
        "max": round(float(series.max()), 6),
        "mean": round(float(series.mean()), 6),
        "median": round(float(series.median()), 6),
        "p25": round(float(series.quantile(0.25)), 6),
        "p75": round(float(series.quantile(0.75)), 6),
        "p95": round(float(series.quantile(0.95)), 6),
    }


def _write_json(payload: dict[str, Any], output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _write_csv(rows: list[dict[str, Any]], output_path: Path, *, overwrite: bool, fieldnames: list[str] | None = None) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _feature_schema_payload(feature_schema) -> dict[str, Any]:
    payload = feature_schema.to_safe_dict()
    payload.update(
        {
            "feature_extraction_version": SPEECH_FEATURE_SCHEMA_VERSION,
            "corpus_metadata_exclusions": ["corpus_name", "source_file", "speaker_id", "safe_speaker_key"],
            "target_label": "canonical_emotion_label",
            "speaker_key_handling": "safe_speaker_key is for leakage-aware splitting only and is excluded from predictive features by default",
            "non_clinical_status": "Speech emotion features are not depression, suicide-risk, crisis, alert, or treatment features.",
            "units": {feature["name"]: feature["description"].split("units: ")[-1].split(";")[0] for feature in payload["features"]},
        }
    )
    return payload


def _canonical_row(record: SpeechCanonicalRecord) -> dict[str, Any]:
    metadata = record.metadata
    return {
        "record_id": record.record_id,
        "corpus_name": record.corpus_name,
        "safe_speaker_key": record.safe_speaker_key,
        "canonical_emotion_label": record.canonical_emotion_label,
        "audio_relative_path": record.audio_relative_path,
        "original_audio_hash": record.original_audio_hash,
        "duration_seconds": metadata.duration_seconds,
        "sample_rate": metadata.sample_rate,
        "channel_count": metadata.channel_count,
        "sample_width": metadata.sample_width,
        "frame_count": metadata.frame_count,
        "file_format": metadata.file_format,
        "file_size_bytes": metadata.file_size_bytes,
        "readable": metadata.readable,
        "validation_warnings": "|".join(record.validation_warnings + metadata.validation_warnings),
    }


def _feature_row(record: SpeechFeatureRecord) -> dict[str, Any]:
    row = {
        "record_id": record.record_id,
        "safe_speaker_key": record.safe_speaker_key,
        "corpus_name": record.corpus_name,
        "canonical_emotion_label": record.canonical_emotion_label,
        "feature_extraction_warnings": "|".join(record.feature_extraction_warnings),
    }
    row.update(record.feature_values)
    return row


def _fingerprint_for_corpus(corpus_name: str, fingerprints: dict[str, DatasetFingerprint | str] | None) -> str:
    if not fingerprints or corpus_name not in fingerprints:
        return "0" * 64
    value = fingerprints[corpus_name]
    return value.combined_sha256 if isinstance(value, DatasetFingerprint) else str(value)


def _build_duplicate_manifest(canonical_records: list[SpeechCanonicalRecord]) -> dict[str, Any]:
    groups: dict[str, list[str]] = defaultdict(list)
    corpus_groups: dict[str, set[str]] = defaultdict(set)
    for record in canonical_records:
        groups[record.original_audio_hash].append(record.record_id)
        corpus_groups[record.original_audio_hash].add(record.corpus_name)
    duplicate_groups = {
        digest: sorted(record_ids)
        for digest, record_ids in sorted(groups.items())
        if len(record_ids) > 1
    }
    cross = {
        digest: sorted(record_ids)
        for digest, record_ids in duplicate_groups.items()
        if len(corpus_groups[digest]) > 1
    }
    return {
        "duplicate_audio_hash_groups": duplicate_groups,
        "duplicate_audio_hash_group_count": len(duplicate_groups),
        "cross_corpus_duplicate_audio_hash_groups": cross,
        "cross_corpus_duplicate_audio_hash_group_count": len(cross),
        "privacy_note": "Duplicate manifests contain record IDs only, not raw speaker IDs or source filenames.",
    }


def _write_normalized_audio(
    path: Path,
    record: SpeechCanonicalRecord,
    *,
    target_sample_rate: int | None,
    mono: bool,
    overwrite: bool,
) -> dict[str, Any]:
    generated_audio_root = paths.get_generated_preprocessing_root() / "speech" / "v1" / "audio"
    sample_rate, audio = load_wav_audio(path, mono=mono, target_sample_rate=target_sample_rate)
    target = generated_audio_root / f"{record.record_id}.wav"
    write_generated_wav(target, sample_rate, audio, overwrite=overwrite)
    return {
        "record_id": record.record_id,
        "generated_audio_relative_path": _relative_to_repo(target),
        "source_audio_hash": record.original_audio_hash,
        "sample_rate": sample_rate,
        "mono": mono,
        "transformations": {
            "convert_to_mono": bool(mono),
            "target_sample_rate": target_sample_rate,
            "augmentation": "none",
            "transcription": "none",
        },
    }


def preprocess_speech_dataset(
    corpus_mapping_config: SpeechCorpusMappingConfig,
    label_mapping_config: SpeechLabelMappingConfig,
    *,
    source_fingerprints: dict[str, DatasetFingerprint | str] | None = None,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_files: int | None = None,
    corpus_filter: list[str] | None = None,
    extract_features: bool = False,
    write_normalized_audio: bool = False,
    target_sample_rate: int | None = None,
    mono: bool = True,
) -> dict[str, Any]:
    selected = set(corpus_filter or [])
    canonical_records: list[SpeechCanonicalRecord] = []
    feature_records: list[SpeechFeatureRecord] = []
    corrupt_files: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    normalized_audio_manifest: list[dict[str, Any]] = []
    source_file_count = 0

    for corpus in corpus_mapping_config.corpora:
        if not corpus.included:
            continue
        if selected and corpus.corpus_name not in selected:
            continue
        files = discover_speech_files(corpus, max_files=max_files)
        source_file_count += len(files)
        source_fingerprint = _fingerprint_for_corpus(corpus.corpus_name, source_fingerprints)
        for path in files:
            try:
                source = parse_speech_file(path, corpus)
                canonical_label = normalize_emotion_label_for_record(source, label_mapping_config)
            except Exception as exc:
                excluded.append({"corpus_name": corpus.corpus_name, "reason": f"parse_or_mapping_error:{exc.__class__.__name__}"})
                continue
            relative_path = _relative_to_repo(path)
            metadata = read_audio_metadata(path)
            original_hash = audio_sha256(path) if path.exists() and path.stat().st_size > 0 else hashlib.sha256(b"").hexdigest()
            warnings = []
            if not metadata.readable:
                warnings.append("unreadable audio retained in report but not silently ignored")
                corrupt_files.append({"record_path_hash": hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16], "corpus_name": corpus.corpus_name, "warnings": metadata.validation_warnings})
            record = SpeechCanonicalRecord(
                record_id=generate_speech_record_id(corpus.corpus_name, relative_path, source_fingerprint),
                corpus_name=corpus.corpus_name,
                safe_speaker_key=generate_safe_speaker_key(corpus.corpus_name, source.speaker_id, source_fingerprint),
                canonical_emotion_label=canonical_label,
                audio_relative_path=relative_path,
                original_audio_hash=original_hash,
                metadata=metadata,
                validation_warnings=warnings,
            )
            canonical_records.append(record)
            if extract_features:
                feature_values, feature_warnings = extract_speech_features(path)
                feature_records.append(
                    SpeechFeatureRecord(
                        record_id=record.record_id,
                        safe_speaker_key=record.safe_speaker_key,
                        corpus_name=record.corpus_name,
                        canonical_emotion_label=record.canonical_emotion_label,
                        feature_values=feature_values,
                        feature_extraction_warnings=feature_warnings,
                    )
                )
            if write_normalized_audio and metadata.readable:
                normalized_audio_manifest.append(
                    _write_normalized_audio(path, record, target_sample_rate=target_sample_rate, mono=mono, overwrite=overwrite)
                )

    canonical_records.sort(key=lambda item: (item.corpus_name, item.audio_relative_path, item.record_id))
    feature_records.sort(key=lambda item: (item.corpus_name, item.record_id))
    validate_preprocessed_speech(canonical_records, feature_records)

    duplicate_manifest = _build_duplicate_manifest(canonical_records)
    durations = [record.metadata.duration_seconds for record in canonical_records if record.metadata.readable]
    corpus_distribution = dict(sorted(Counter(record.corpus_name for record in canonical_records).items()))
    label_distribution = dict(sorted(Counter(record.canonical_emotion_label for record in canonical_records).items()))
    sample_rates = dict(sorted(Counter(str(record.metadata.sample_rate) for record in canonical_records if record.metadata.readable).items()))
    channels = dict(sorted(Counter(str(record.metadata.channel_count) for record in canonical_records if record.metadata.readable).items()))
    missing_summary = {name: 0 for name in FEATURE_COLUMNS}
    if extract_features:
        for name in FEATURE_COLUMNS:
            missing_summary[name] = int(sum(1 for record in feature_records if name not in record.feature_values))
    else:
        missing_summary = {name: len(canonical_records) for name in FEATURE_COLUMNS}
    speaker_leakage = detect_speaker_leakage_risk(canonical_records)
    fingerprint_map = {
        corpus.corpus_name: _fingerprint_for_corpus(corpus.corpus_name, source_fingerprints)
        for corpus in corpus_mapping_config.corpora
        if corpus.included and (not selected or corpus.corpus_name in selected)
    }
    report = SpeechPreprocessingReport(
        preprocessing_version=SPEECH_PREPROCESSING_VERSION,
        feature_schema_version=SPEECH_FEATURE_SCHEMA_VERSION,
        label_mapping_version=SPEECH_LABEL_MAPPING_VERSION,
        corpus_mapping_version=SPEECH_CORPUS_MAPPING_VERSION,
        source_fingerprints=fingerprint_map,
        source_file_count=source_file_count,
        readable_file_count=sum(1 for record in canonical_records if record.metadata.readable),
        unreadable_file_count=sum(1 for record in canonical_records if not record.metadata.readable),
        output_record_count=len(canonical_records),
        excluded_record_count=len(excluded),
        corpus_distribution=corpus_distribution,
        label_distribution=label_distribution,
        speaker_count=int(len({record.safe_speaker_key for record in canonical_records})),
        sample_rate_distribution=sample_rates,
        channel_distribution=channels,
        duration_summary=_duration_summary(durations),
        duplicate_summary={
            "duplicate_audio_hash_group_count": duplicate_manifest["duplicate_audio_hash_group_count"],
            "cross_corpus_duplicate_audio_hash_group_count": duplicate_manifest["cross_corpus_duplicate_audio_hash_group_count"],
            "speaker_leakage_risk": speaker_leakage,
        },
        feature_missing_summary=missing_summary,
        warnings=[
            "Speech-emotion recognition is not equivalent to depression detection or suicide-risk prediction.",
            "Acted emotional speech may not represent natural distress or Sri Lankan student recordings.",
            "Voice is biometric and highly sensitive; generated acoustic features remain sensitive.",
            "Corpus identity can become a shortcut feature and is excluded from predictive features by default.",
            "Speaker-independent splitting is required later; no train/validation/test split was created here.",
            "No transcription, model download, model training, alerting, or treatment recommendation occurred.",
            "No production voice records, PostgreSQL data, API routes, migrations, or SafeTalk logic were changed.",
        ],
    )
    feature_schema = build_speech_feature_schema()

    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Speech preprocessing outputs must be under generated/")
        canonical_rows = [_canonical_row(record) for record in canonical_records]
        feature_rows = [_feature_row(record) for record in feature_records]
        feature_fieldnames = ["record_id", "safe_speaker_key", "corpus_name", "canonical_emotion_label", *FEATURE_COLUMNS, "feature_extraction_warnings"]
        if not extract_features:
            feature_rows = []
        corpus_summary = {
            corpus: {
                "record_count": int(count),
                "safe_speaker_count": int(len({record.safe_speaker_key for record in canonical_records if record.corpus_name == corpus})),
            }
            for corpus, count in corpus_distribution.items()
        }
        outputs = {
            "canonical_manifest_csv": str(_write_csv(canonical_rows, resolved_output_dir / "speech_canonical_manifest.csv", overwrite=overwrite)),
            "features_csv": str(_write_csv(feature_rows, resolved_output_dir / "speech_features.csv", overwrite=overwrite, fieldnames=feature_fieldnames)),
            "feature_schema_json": str(_write_json(_feature_schema_payload(feature_schema), resolved_output_dir / "speech_feature_schema.json", overwrite=overwrite)),
            "report_json": str(_write_json(report.to_safe_dict(), resolved_output_dir / "speech_preprocessing_report.json", overwrite=overwrite)),
            "record_manifest_json": str(
                _write_json(
                    {
                        "dataset": "speech-emotion",
                        "record_count": len(canonical_records),
                        "record_id_strategy": "hash(corpus, source-relative path, corpus source fingerprint); raw speaker IDs excluded",
                        "record_ids": [record.record_id for record in canonical_records],
                    },
                    resolved_output_dir / "speech_record_manifest.json",
                    overwrite=overwrite,
                )
            ),
            "corrupt_files_json": str(_write_json({"corrupt_or_unreadable_files": corrupt_files}, resolved_output_dir / "speech_corrupt_files.json", overwrite=overwrite)),
            "duplicate_manifest_json": str(_write_json(duplicate_manifest, resolved_output_dir / "speech_duplicate_manifest.json", overwrite=overwrite)),
            "corpus_summary_json": str(_write_json(corpus_summary, resolved_output_dir / "speech_corpus_summary.json", overwrite=overwrite)),
            "label_distribution_json": str(_write_json(label_distribution, resolved_output_dir / "speech_label_distribution.json", overwrite=overwrite)),
        }
        md_path = resolved_output_dir / "speech_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_speech_preprocessing_markdown(report), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)
        if write_normalized_audio:
            outputs["normalized_audio_manifest_json"] = str(
                _write_json({"normalized_audio": normalized_audio_manifest}, resolved_output_dir / "speech_normalized_audio_manifest.json", overwrite=overwrite)
            )

    return {
        "valid": True,
        "validate_only": validate_only,
        "extract_features": extract_features,
        "source_files": source_file_count,
        "output_records": len(canonical_records),
        "excluded_records": len(excluded),
        "readable_files": report.readable_file_count,
        "unreadable_files": report.unreadable_file_count,
        "corpus_distribution": corpus_distribution,
        "label_distribution": label_distribution,
        "sample_rate_distribution": sample_rates,
        "channel_distribution": channels,
        "duration_summary": report.duration_summary,
        "duplicate_summary": report.duplicate_summary,
        "feature_missing_summary": missing_summary,
        "feature_columns": list(FEATURE_COLUMNS),
        "report": report,
        "feature_schema": feature_schema,
        "outputs": outputs,
    }


def validate_preprocessed_speech(canonical_records: list[SpeechCanonicalRecord], feature_records: list[SpeechFeatureRecord] | None = None) -> None:
    for record in canonical_records:
        if record.safe_speaker_key.startswith(record.corpus_name):
            raise ValueError("safe speaker keys must not expose raw speaker identifiers")
        if Path(record.audio_relative_path).is_absolute():
            raise ValueError("speech reports and manifests must use relative paths")
    if feature_records:
        validate_feature_values(feature_records)
        blocked = {"corpus_name", "source_file", "speaker_id", "safe_speaker_key"}
        if blocked & set(FEATURE_COLUMNS):
            raise ValueError("Speech feature columns contain metadata leakage")

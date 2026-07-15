"""Validation helpers for speech preprocessing."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from app.ml.preprocessing.speech.audio_io import audio_sha256, extract_audio_metadata
from app.ml.preprocessing.speech.constants import OPTIONAL_AUDIO_EXTENSIONS, SUPPORTED_AUDIO_EXTENSIONS
from app.ml.preprocessing.speech.filename_parsers import parse_with_named_parser
from app.ml.preprocessing.speech.mapping import normalize_emotion_label
from app.ml.preprocessing.speech.schemas import SpeechCanonicalRecord, SpeechCorpusMappingConfig, SpeechFeatureRecord, SpeechLabelMappingConfig


def validate_speech_source_paths(corpus_mapping: SpeechCorpusMappingConfig, *, repository_root: Path | None = None) -> dict[str, str]:
    root = repository_root or Path.cwd()
    missing: dict[str, str] = {}
    for corpus in corpus_mapping.corpora:
        path = Path(corpus.source_path)
        resolved = path if path.is_absolute() else root / path
        if not resolved.exists():
            missing[corpus.corpus_name] = corpus.source_path
    if missing:
        raise FileNotFoundError(f"Missing speech source paths: {missing}")
    return {corpus.corpus_name: corpus.source_path for corpus in corpus_mapping.corpora}


def validate_audio_extensions(files: Iterable[Path]) -> dict[str, int]:
    extensions = Counter(path.suffix.lower() or "<none>" for path in files)
    unsupported = sorted(ext for ext in extensions if ext not in SUPPORTED_AUDIO_EXTENSIONS and ext not in OPTIONAL_AUDIO_EXTENSIONS)
    if unsupported:
        raise ValueError(f"Unsupported speech audio extensions: {unsupported}")
    return dict(sorted(extensions.items()))


def validate_filename_parsing(files: Iterable[Path], parser_name: str) -> dict[str, int | list[str]]:
    malformed: list[str] = []
    parsed = 0
    for path in files:
        try:
            parse_with_named_parser(parser_name, path)
            parsed += 1
        except ValueError:
            malformed.append(path.name)
    return {"parsed_count": parsed, "malformed_count": len(malformed), "malformed_examples": malformed[:25]}


def validate_emotion_labels(records, label_mapping: SpeechLabelMappingConfig) -> dict[str, int]:
    unsupported: Counter[str] = Counter()
    for record in records:
        try:
            normalize_emotion_label(record.corpus_name, record.original_emotion_label, label_mapping)
        except ValueError:
            unsupported[f"{record.corpus_name}:{record.original_emotion_label}"] += 1
    if unsupported:
        raise ValueError(f"Unsupported speech emotion labels: {dict(sorted(unsupported.items()))}")
    return {}


def validate_speaker_ids(records) -> dict[str, int]:
    missing = Counter(record.corpus_name for record in records if not str(record.speaker_id).strip())
    if missing:
        raise ValueError(f"Missing speech speaker IDs: {dict(sorted(missing.items()))}")
    return dict(sorted(Counter(record.corpus_name for record in records).items()))


def detect_audio_duplicates(paths: Iterable[Path]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            groups[audio_sha256(path)].append(path.name)
    return {digest: sorted(names) for digest, names in sorted(groups.items()) if len(names) > 1}


def detect_cross_corpus_duplicates(corpus_paths: dict[str, list[Path]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for corpus, paths in corpus_paths.items():
        for path in paths:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                groups[audio_sha256(path)].append(f"{corpus}/{path.name}")
    return {digest: sorted(names) for digest, names in sorted(groups.items()) if len({item.split('/')[0] for item in names}) > 1}


def detect_corrupt_audio(paths: Iterable[Path]) -> list[str]:
    corrupt = []
    for path in paths:
        metadata = extract_audio_metadata(path)
        if not metadata.readable:
            corrupt.append(path.name)
    return corrupt


def detect_short_audio(records: Iterable[SpeechCanonicalRecord], *, minimum_seconds: float = 0.25) -> list[str]:
    return [record.record_id for record in records if record.metadata.readable and record.metadata.duration_seconds < minimum_seconds]


def detect_long_audio(records: Iterable[SpeechCanonicalRecord], *, maximum_seconds: float = 30.0) -> list[str]:
    return [record.record_id for record in records if record.metadata.readable and record.metadata.duration_seconds > maximum_seconds]


def detect_sample_rate_outliers(records: Iterable[SpeechCanonicalRecord], expected_by_corpus: dict[str, int | None]) -> dict[str, int]:
    outliers: Counter[str] = Counter()
    for record in records:
        expected = expected_by_corpus.get(record.corpus_name)
        if expected and record.metadata.readable and record.metadata.sample_rate != expected:
            outliers[record.corpus_name] += 1
    return dict(sorted(outliers.items()))


def detect_channel_outliers(records: Iterable[SpeechCanonicalRecord], *, expected_channels: int = 1) -> dict[str, int]:
    outliers: Counter[str] = Counter()
    for record in records:
        if record.metadata.readable and record.metadata.channel_count != expected_channels:
            outliers[str(record.metadata.channel_count)] += 1
    return dict(sorted(outliers.items()))


def validate_feature_values(records: Iterable[SpeechFeatureRecord]) -> None:
    for record in records:
        for name, value in record.feature_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"Speech feature contains NaN or infinity: {record.record_id}:{name}")


def detect_speaker_leakage_risk(records: Iterable[SpeechCanonicalRecord]) -> dict[str, object]:
    by_speaker: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        by_speaker[record.safe_speaker_key][record.canonical_emotion_label] += 1
    multi_record_speakers = sum(1 for labels in by_speaker.values() if sum(labels.values()) > 1)
    return {
        "speaker_independent_splitting_possible": len(by_speaker) >= 2,
        "speaker_count": len(by_speaker),
        "multi_record_speaker_count": multi_record_speakers,
        "risk": "high" if multi_record_speakers else "low",
        "notes": "Later train/test splits must group by safe_speaker_key to avoid speaker leakage.",
    }


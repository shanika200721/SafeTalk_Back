"""Read-only audio dataset audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import wave
from pathlib import Path

import pandas as pd

from app.ml.audit.base import AuditContext, build_report, issue
from app.ml.audit.schemas import AudioAuditResult, AuditIssue, AuditSeverity, ClassDistributionItem, LengthSummary
from app.ml.common.hashing import sha256_file


_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}


def _summary(values: list[float]) -> LengthSummary:
    if not values:
        return LengthSummary()
    series = pd.Series(values, dtype="float64")
    return LengthSummary(
        minimum=float(series.min()),
        maximum=float(series.max()),
        mean=float(series.mean()),
        median=float(series.median()),
        percentile_25=float(series.quantile(0.25)),
        percentile_75=float(series.quantile(0.75)),
        percentile_95=float(series.quantile(0.95)),
    )


def _iter_audio_files(source_path: Path, max_files: int) -> tuple[list[Path], bool]:
    if source_path.is_file():
        return [source_path], False
    files = sorted(path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in _AUDIO_EXTENSIONS)
    sampled = max_files > 0 and len(files) > max_files
    return files[:max_files] if sampled else files, sampled


def _label_for(path: Path, source_path: Path, folder_label_depth: int) -> str | None:
    if not source_path.is_dir():
        return None
    relative = path.relative_to(source_path)
    if len(relative.parts) <= folder_label_depth:
        return None
    return relative.parts[-(folder_label_depth + 1)]


def _distribution(labels: Counter[str], total: int) -> dict[str, list[ClassDistributionItem]]:
    values = [
        ClassDistributionItem(label=label, count=count, percentage=round((count / max(total, 1)) * 100, 4))
        for label, count in sorted(labels.items())
    ]
    return {"folder_label": values} if values else {}


def _read_wav(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        duration = frames / float(sample_rate) if sample_rate else 0.0
    return duration, sample_rate, channels


def audit_audio_dataset(context: AuditContext):
    files, sampled = _iter_audio_files(context.source_path, context.options.max_files)
    issues: list[AuditIssue] = []
    if sampled:
        issues.append(
            issue(
                "file_sampling_used",
                AuditSeverity.INFO,
                "Audio audit used deterministic file limit",
                count=len(files),
                details={"max_files": context.options.max_files},
            )
        )

    durations: list[float] = []
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    unreadable = 0
    corrupt = 0
    empty = 0
    hashes: dict[str, list[str]] = defaultdict(list)

    allow_outside = getattr(context.dataset_config, "validation_context", None) == "test"
    for path in files:
        formats[path.suffix.lower().lstrip(".") or "<none>"] += 1
        if path.stat().st_size == 0:
            empty += 1
            unreadable += 1
            issues.append(issue("empty_audio_file", AuditSeverity.WARNING, "Zero-byte audio file found", count=1))
            continue
        label = _label_for(path, context.source_path, context.options.folder_label_depth)
        if label:
            labels[label] += 1
        hashes[sha256_file(path, allow_outside_project=allow_outside)].append(path.name)
        if path.suffix.lower() != ".wav":
            unreadable += 1
            issues.append(
                issue(
                    "unsupported_audio_metadata",
                    AuditSeverity.WARNING,
                    "Only WAV metadata is supported without optional audio dependencies",
                    details={"extension": path.suffix.lower()},
                )
            )
            continue
        try:
            duration, sample_rate, channel_count = _read_wav(path)
        except Exception:
            unreadable += 1
            corrupt += 1
            issues.append(issue("corrupt_audio_file", AuditSeverity.WARNING, "Unreadable or corrupt WAV file found", count=1))
            continue
        durations.append(duration)
        sample_rates[str(sample_rate)] += 1
        channels[str(channel_count)] += 1
        if context.options.minimum_audio_duration is not None and duration < context.options.minimum_audio_duration:
            issues.append(issue("audio_too_short", AuditSeverity.WARNING, "Audio shorter than configured minimum", count=1))
        if context.options.maximum_audio_duration is not None and duration > context.options.maximum_audio_duration:
            issues.append(issue("audio_too_long", AuditSeverity.WARNING, "Audio longer than configured maximum", count=1))

    duplicate_hash_groups = sum(1 for grouped in hashes.values() if len(grouped) > 1)
    if duplicate_hash_groups:
        issues.append(
            issue(
                "duplicate_audio_hashes",
                AuditSeverity.WARNING,
                "Exact duplicate audio file hashes detected",
                count=duplicate_hash_groups,
            )
        )

    result = AudioAuditResult(
        file_count=len(files),
        readable_file_count=len(durations),
        unreadable_file_count=unreadable,
        total_duration_seconds=round(float(sum(durations)), 6),
        duration_summary=_summary(durations),
        sample_rate_distribution=dict(sorted(sample_rates.items())),
        channel_distribution=dict(sorted(channels.items())),
        format_distribution=dict(sorted(formats.items())),
        empty_file_count=empty,
        corrupt_file_count=corrupt,
        duplicate_hash_group_count=duplicate_hash_groups,
        label_distribution=_distribution(labels, sum(labels.values())),
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
    )
    return build_report(
        context,
        modality_result_name="audio_result",
        modality_result=result,
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
        notes="Audio audit inspected container metadata only; no transcription or feature extraction ran.",
    )

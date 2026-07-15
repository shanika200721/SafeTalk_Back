"""Read-only image dataset audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

from app.ml.audit.base import AuditContext, build_report, issue
from app.ml.audit.schemas import AuditIssue, AuditSeverity, ClassDistributionItem, ImageAuditResult, LengthSummary
from app.ml.common.hashing import sha256_file


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


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


def _iter_image_files(source_path: Path, max_files: int) -> tuple[list[Path], bool]:
    if source_path.is_file():
        return [source_path], False
    files = sorted(path for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS)
    sampled = max_files > 0 and len(files) > max_files
    return files[:max_files] if sampled else files, sampled


def _label_for(path: Path, source_path: Path, folder_label_depth: int) -> str | None:
    if not source_path.is_dir():
        return None
    relative = path.relative_to(source_path)
    if len(relative.parts) <= folder_label_depth:
        return None
    return relative.parts[-(folder_label_depth + 1)]


def _split_for(path: Path, source_path: Path, split_names: tuple[str, ...]) -> str | None:
    if not source_path.is_dir():
        return None
    parts = path.relative_to(source_path).parts[:-1]
    lowered = {name.lower(): name for name in split_names}
    for part in parts:
        if part.lower() in lowered:
            return lowered[part.lower()]
    return None


def _distribution(labels: Counter[str], total: int) -> dict[str, list[ClassDistributionItem]]:
    values = [
        ClassDistributionItem(label=label, count=count, percentage=round((count / max(total, 1)) * 100, 4))
        for label, count in sorted(labels.items())
    ]
    return {"folder_label": values} if values else {}


def audit_image_dataset(context: AuditContext):
    files, sampled = _iter_image_files(context.source_path, context.options.max_files)
    issues: list[AuditIssue] = []
    if sampled:
        issues.append(
            issue(
                "file_sampling_used",
                AuditSeverity.INFO,
                "Image audit used deterministic file limit",
                count=len(files),
                details={"max_files": context.options.max_files},
            )
        )

    widths: list[float] = []
    heights: list[float] = []
    modes: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    split_hashes: dict[str, set[str]] = defaultdict(set)
    unreadable = 0
    corrupt = 0

    allow_outside = getattr(context.dataset_config, "validation_context", None) == "test"
    for path in files:
        formats[path.suffix.lower().lstrip(".") or "<none>"] += 1
        if path.stat().st_size == 0:
            unreadable += 1
            corrupt += 1
            issues.append(issue("empty_image_file", AuditSeverity.WARNING, "Zero-byte image file found", count=1))
            continue
        digest = sha256_file(path, allow_outside_project=allow_outside)
        hashes[digest].append(path.name)
        split = _split_for(path, context.source_path, context.options.predefined_split_folder_names)
        if split:
            split_hashes[split].add(digest)
        label = _label_for(path, context.source_path, context.options.folder_label_depth)
        if label:
            labels[label] += 1
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                widths.append(float(image.width))
                heights.append(float(image.height))
                modes[image.mode] += 1
                if image.format:
                    formats[image.format.lower()] += 0
        except Exception:
            unreadable += 1
            corrupt += 1
            issues.append(issue("corrupt_image_file", AuditSeverity.WARNING, "Unreadable or corrupt image file found", count=1))

    duplicate_hash_groups = sum(1 for grouped in hashes.values() if len(grouped) > 1)
    if duplicate_hash_groups:
        issues.append(
            issue(
                "duplicate_image_hashes",
                AuditSeverity.WARNING,
                "Exact duplicate image file hashes detected",
                count=duplicate_hash_groups,
            )
        )
    if "train" in split_hashes and "test" in split_hashes:
        overlap = len(split_hashes["train"] & split_hashes["test"])
        if overlap:
            issues.append(
                issue(
                    "train_test_duplicate_hash_overlap",
                    AuditSeverity.WARNING,
                    "Duplicate image hashes appear across predefined train/test folders",
                    count=overlap,
                )
            )

    result = ImageAuditResult(
        file_count=len(files),
        readable_file_count=len(widths),
        unreadable_file_count=unreadable,
        width_summary=_summary(widths),
        height_summary=_summary(heights),
        color_mode_distribution=dict(sorted(modes.items())),
        format_distribution=dict(sorted(formats.items())),
        corrupt_file_count=corrupt,
        duplicate_hash_group_count=duplicate_hash_groups,
        label_distribution=_distribution(labels, sum(labels.values())),
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
    )
    return build_report(
        context,
        modality_result_name="image_result",
        modality_result=result,
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
        notes="Image audit inspected metadata only; no resizing, face detection, or embeddings ran.",
    )

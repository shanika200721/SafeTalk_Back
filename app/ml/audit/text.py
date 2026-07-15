"""Privacy-safe read-only text dataset audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
import unicodedata

import pandas as pd

from app.ml.audit.base import AuditContext, build_report, issue, safe_label_value
from app.ml.audit.schemas import AuditIssue, AuditSeverity, ClassDistributionItem, LengthSummary, TextAuditResult
from app.ml.audit.tabular import load_tabular_source


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.I)
_USERNAME_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]{3,}")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{7,}\d)")
_PERSON_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def _summary(values: list[int]) -> LengthSummary:
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


def _distribution(df: pd.DataFrame, label_columns: list[str]) -> dict[str, list[ClassDistributionItem]]:
    result: dict[str, list[ClassDistributionItem]] = {}
    for column in label_columns:
        if column not in df.columns:
            continue
        total = int(df[column].notna().sum())
        values = []
        for label, count in df[column].dropna().value_counts().head(25).items():
            values.append(
                ClassDistributionItem(
                    label=safe_label_value(label, field_name=column),
                    count=int(count),
                    percentage=round((int(count) / max(total, 1)) * 100, 4),
                )
            )
        result[column] = values
    return result


def _language_distribution(texts: list[str]) -> dict[str, int]:
    counts = Counter()
    for text in texts:
        if not text:
            counts["blank"] += 1
            continue
        ascii_letters = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        if non_ascii > max(3, len(text) * 0.05):
            counts["non_ascii_or_mixed"] += 1
        elif ascii_letters >= max(5, len(text.replace(" ", "")) * 0.5):
            counts["english_like"] += 1
        else:
            counts["latin_ascii_unknown"] += 1
    return dict(sorted(counts.items()))


def _near_duplicate_candidates(normalized_texts: list[str], limit: int) -> int:
    signatures: Counter[str] = Counter()
    checked = 0
    for text in normalized_texts:
        if not text:
            continue
        checked += 1
        if checked > limit:
            break
        tokens = text.split()
        if len(tokens) < 8:
            signature = text[:80]
        else:
            signature = " ".join(tokens[:4] + tokens[-4:])
        signatures[signature] += 1
    return sum(count - 1 for count in signatures.values() if count > 1)


def _label_columns(context: AuditContext, df: pd.DataFrame) -> list[str]:
    labels = list(context.dataset_config.label_columns)
    if context.options.label_column and context.options.label_column not in labels:
        labels.append(context.options.label_column)
    return [column for column in labels if column in df.columns]


def audit_text_dataset(context: AuditContext):
    df = load_tabular_source(context.source_path, context.dataset_config.file_format)
    issues: list[AuditIssue] = []
    original_count = len(df)
    if context.options.max_records and len(df) > context.options.max_records:
        df = df.sample(n=context.options.max_records, random_state=context.options.sample_seed).sort_index()
        issues.append(
            issue(
                "sampling_used",
                AuditSeverity.INFO,
                "Audit used deterministic row sampling",
                count=len(df),
                details={"source_record_count": original_count, "sample_seed": context.options.sample_seed},
            )
        )

    text_column = context.options.text_column
    if not text_column:
        configured_text = [column for column in context.dataset_config.feature_columns if column in df.columns]
        if len(configured_text) == 1:
            text_column = configured_text[0]
    if not text_column or text_column not in df.columns:
        raise ValueError("Text audit requires an explicit text column in DatasetConfig feature_columns or --text-column")

    normalized = [_normalize_text(value) for value in df[text_column].tolist()]
    missing_text_count = sum(1 for item in normalized if not item)
    if missing_text_count:
        issues.append(
            issue(
                "missing_or_blank_text",
                AuditSeverity.WARNING,
                "Text column contains missing or blank records",
                field_name=text_column,
                count=missing_text_count,
            )
        )

    duplicate_text_count = int(pd.Series(normalized).duplicated().sum())
    label_columns = _label_columns(context, df)
    conflicting = 0
    if label_columns:
        labels = df[label_columns[0]].astype(str).tolist()
        groups: dict[str, set[str]] = defaultdict(set)
        for text, label in zip(normalized, labels):
            if text:
                groups[text].add(label)
        conflicting = sum(1 for labels_for_text in groups.values() if len(labels_for_text) > 1)
        if conflicting:
            issues.append(
                issue(
                    "duplicate_text_conflicting_labels",
                    AuditSeverity.WARNING,
                    "Duplicate normalized text appears with conflicting labels",
                    count=conflicting,
                )
            )

    chars = [len(text) for text in normalized if text]
    words = [len(text.split()) for text in normalized if text]
    pii_counts = {
        "url": sum(len(_URL_RE.findall(str(value))) for value in df[text_column].fillna("").tolist()),
        "email": sum(len(_EMAIL_RE.findall(str(value))) for value in df[text_column].fillna("").tolist()),
        "username": sum(len(_USERNAME_RE.findall(str(value))) for value in df[text_column].fillna("").tolist()),
        "phone": sum(len(_PHONE_RE.findall(str(value))) for value in df[text_column].fillna("").tolist()),
        "person_name": sum(len(_PERSON_NAME_RE.findall(str(value))) for value in df[text_column].fillna("").tolist()),
    }
    for key, count in pii_counts.items():
        if count:
            issues.append(
                issue(
                    f"{key}_pattern_detected",
                    AuditSeverity.WARNING,
                    f"Text contains {key.replace('_', ' ')} patterns",
                    field_name=text_column,
                    count=count,
                )
            )

    near_limit = min(len(normalized), max(1000, context.options.max_records))
    near_duplicates = _near_duplicate_candidates(normalized, near_limit)
    if len(normalized) > near_limit:
        issues.append(
            issue(
                "near_duplicate_bounded",
                AuditSeverity.INFO,
                "Near-duplicate detection used deterministic bounded comparison",
                count=near_limit,
            )
        )

    result = TextAuditResult(
        record_count=int(len(df)),
        label_distribution=_distribution(df, label_columns),
        missing_text_count=missing_text_count,
        exact_duplicate_text_count=duplicate_text_count,
        duplicate_text_conflicting_labels_count=conflicting,
        near_duplicate_candidate_count=near_duplicates,
        character_length_summary=_summary(chars),
        word_count_summary=_summary(words),
        url_occurrence_count=pii_counts["url"],
        email_occurrence_count=pii_counts["email"],
        username_occurrence_count=pii_counts["username"],
        phone_occurrence_count=pii_counts["phone"],
        possible_person_name_occurrence_count=pii_counts["person_name"],
        language_distribution=_language_distribution(normalized[: min(len(normalized), 5000)]),
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
    )
    return build_report(
        context,
        modality_result_name="text_result",
        modality_result=result,
        issues=sorted(issues, key=lambda item: (item.code, item.field_name or "")),
        notes="Text audit used normalized text only for comparison; raw text was not stored.",
    )

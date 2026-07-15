"""Canonical text preprocessing without model training or feature fitting."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from app.ml.common import paths
from app.ml.common.schemas import DatasetConfig, DatasetFingerprint, PreprocessingConfig
from app.ml.preprocessing.text.constants import (
    CANONICAL_COLUMNS,
    DATASET_NAME,
    DATASET_VERSION,
    LABEL_COLUMN,
    PRIVACY_PLACEHOLDERS,
    RECORD_ID_PREFIX,
    TEXT_COLUMN,
    TEXT_FEATURE_SCHEMA_VERSION,
    TEXT_LABEL_MAPPING_VERSION,
    TEXT_PREPROCESSING_VERSION,
    TEXT_PRIVACY_RULESET_VERSION,
)
from app.ml.preprocessing.text.duplicates import apply_duplicate_policy, bounded_near_duplicate_candidates, exact_duplicate_groups, sha256_text
from app.ml.preprocessing.text.features import build_text_feature_schema, default_future_feature_contracts
from app.ml.preprocessing.text.mapping import normalize_label
from app.ml.preprocessing.text.normalization import normalize_text
from app.ml.preprocessing.text.reporting import create_text_preprocessing_markdown
from app.ml.preprocessing.text.schemas import (
    TextLabelMappingConfig,
    TextPreprocessingReport,
    TextPrivacySummary,
    TextSourceSelectionConfig,
)
from app.ml.preprocessing.text.validation import (
    contains_privacy_placeholder,
    detect_empty_text,
    detect_engineered_feature_leakage,
    detect_privacy_pattern_leakage,
    detect_target_leakage_columns,
    placeholder_only_count,
    validate_label_values,
    validate_predefined_test_overlap,
    validate_source_selection,
    validate_text_source_columns,
    validate_text_values,
)


def load_text_source(dataset_config: DatasetConfig) -> pd.DataFrame:
    source_path = dataset_config.validate_source_exists()
    return pd.read_csv(source_path)


def build_text_hash(comparison_text: str) -> str:
    return sha256_text(comparison_text)


def generate_text_record_id(
    *,
    dataset_version: str,
    source_file_identity: str,
    source_row_index: int,
    source_fingerprint: str,
) -> str:
    if source_row_index < 0:
        raise ValueError("source_row_index must be non-negative")
    if not source_fingerprint or len(source_fingerprint) != 64:
        raise ValueError("source_fingerprint must be a SHA-256 hash")
    payload = f"{RECORD_ID_PREFIX}:{dataset_version}:{source_file_identity}:{source_row_index}:{source_fingerprint}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{RECORD_ID_PREFIX}-{source_row_index + 1:06d}-{digest}"


def normalize_text_record(value: object):
    return normalize_text(value)


def canonicalize_text_dataframe(
    df: pd.DataFrame,
    label_mapping_config: TextLabelMappingConfig,
    *,
    source_fingerprint: str,
    source_name: str,
    text_column: str = TEXT_COLUMN,
    label_column: str = LABEL_COLUMN,
    max_records: int | None = None,
) -> pd.DataFrame:
    validate_text_source_columns(df.columns, text_column, label_column)
    validate_text_values(df, text_column)
    validate_label_values(df, label_column, label_mapping_config)
    working = df.head(max_records).copy() if max_records is not None else df.copy()
    rows: list[dict[str, Any]] = []
    for source_position, row in working.reset_index(drop=True).iterrows():
        normalized = normalize_text_record(row[text_column])
        canonical_label = normalize_label(row[label_column], label_mapping_config)
        text_hash = build_text_hash(normalized.comparison_text)
        warnings: list[str] = []
        if not normalized.display_text:
            warnings.append("empty_text_after_normalization")
        if placeholder_only_count([normalized.display_text]):
            warnings.append("text_contains_only_privacy_placeholders")
        rows.append(
            {
                "record_id": generate_text_record_id(
                    dataset_version=DATASET_VERSION,
                    source_file_identity=source_name,
                    source_row_index=int(source_position),
                    source_fingerprint=source_fingerprint,
                ),
                "normalized_text": normalized.display_text,
                "comparison_text": normalized.comparison_text,
                "canonical_label": canonical_label,
                "source_name": source_name,
                "source_row_index": int(source_position),
                "original_id": None if "Unique_ID" not in working.columns or pd.isna(row.get("Unique_ID")) else str(row.get("Unique_ID")),
                "text_hash": text_hash,
                "url_count": normalized.privacy_summary.url_count,
                "email_count": normalized.privacy_summary.email_count,
                "phone_count": normalized.privacy_summary.phone_count,
                "username_count": normalized.privacy_summary.username_count,
                "ip_address_count": normalized.privacy_summary.ip_address_count,
                "community_count": normalized.privacy_summary.community_count,
                "possible_person_identifier_count": normalized.privacy_summary.possible_person_identifier_count,
                "character_count": len(normalized.display_text),
                "word_count": len(normalized.display_text.split()),
                "line_count": 0 if not normalized.display_text else normalized.display_text.count("\n") + 1,
                "placeholder_count": contains_privacy_placeholder(normalized.display_text),
                "validation_warnings": ";".join(warnings),
            }
        )
    canonical = pd.DataFrame(rows)
    validate_preprocessed_text(canonical)
    return canonical


def validate_preprocessed_text(canonical_df: pd.DataFrame) -> None:
    required = set(CANONICAL_COLUMNS) | {"comparison_text", "source_row_index"}
    missing = required - set(canonical_df.columns)
    if missing:
        raise ValueError(f"Canonical text data missing columns: {sorted(missing)}")
    for value in canonical_df["text_hash"].astype(str):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError("Canonical text hash must be SHA-256")
    leakage = detect_privacy_pattern_leakage(canonical_df["normalized_text"].astype(str).tolist())
    if any(leakage.values()):
        raise ValueError(f"Canonical normalized text contains unredacted identifier patterns: {leakage}")
    blocked_features = {"canonical_label", "source_name", "source_row_index", "original_id"}
    if "model_features" in canonical_df.columns and blocked_features & set(canonical_df["model_features"]):
        raise ValueError("Target/source metadata must not be treated as model features")


def _language_summary(texts: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts:
        if not text:
            counts["blank"] += 1
            continue
        ascii_letters = sum(1 for ch in text if "a" <= ch.lower() <= "z")
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        compact = text.replace(" ", "")
        if non_ascii > max(3, len(text) * 0.05):
            counts["non_ascii_or_mixed"] += 1
        elif ascii_letters >= max(5, len(compact) * 0.5):
            counts["english_like"] += 1
        else:
            counts["latin_ascii_unknown"] += 1
    return dict(sorted(counts.items()))


def _length_summary(canonical_df: pd.DataFrame) -> dict[str, Any]:
    if canonical_df.empty:
        return {}
    char = canonical_df["character_count"].astype(float)
    words = canonical_df["word_count"].astype(float)
    return {
        "character_count": {
            "min": float(char.min()),
            "max": float(char.max()),
            "mean": float(char.mean()),
            "median": float(char.median()),
            "p95": float(char.quantile(0.95)),
        },
        "word_count": {
            "min": float(words.min()),
            "max": float(words.max()),
            "mean": float(words.mean()),
            "median": float(words.median()),
            "p95": float(words.quantile(0.95)),
        },
    }


def _privacy_totals(canonical_df: pd.DataFrame) -> dict[str, int]:
    names = ["url_count", "email_count", "phone_count", "username_count", "ip_address_count", "community_count", "possible_person_identifier_count"]
    return {name: int(canonical_df[name].sum()) if name in canonical_df.columns else 0 for name in names}


def _write_json(payload: dict[str, Any], output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    temp_path.replace(output_path)
    return output_path


def _write_csv(df: pd.DataFrame, output_path: Path, *, overwrite: bool) -> Path:
    paths.assert_not_raw_dataset_path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return output_path


def _record_manifest(canonical_df: pd.DataFrame, source_fingerprint: str) -> dict[str, Any]:
    return {
        "dataset": RECORD_ID_PREFIX,
        "source_fingerprint": source_fingerprint,
        "record_count": int(len(canonical_df)),
        "record_id_strategy": "hash(dataset_version,source_file_identity,source_row_index,source_fingerprint); not raw-text-derived",
        "record_ids": canonical_df["record_id"].astype(str).tolist(),
    }


def _duplicate_manifest(groups, near_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "exact_duplicate_groups": [group.to_safe_dict() for group in groups],
        "near_duplicate_candidates": near_candidates,
        "policy": "conflicting exact duplicates are quarantined by default; near duplicates are candidates only",
    }


def _source_overlap_report(overlap: dict[str, Any]) -> dict[str, Any]:
    return {
        "reference_overlap_policy": "reported only; reference test file is not combined with authoritative source",
        **overlap,
    }


def build_canonical_text_table(
    source_df: pd.DataFrame,
    label_mapping_config: TextLabelMappingConfig,
    *,
    source_fingerprint: str,
    source_name: str,
    max_records: int | None = None,
) -> pd.DataFrame:
    return canonicalize_text_dataframe(
        source_df,
        label_mapping_config,
        source_fingerprint=source_fingerprint,
        source_name=source_name,
        max_records=max_records,
    )


def preprocess_text_dataframe(
    source_df: pd.DataFrame,
    preprocessing_config: PreprocessingConfig | None,
    label_mapping_config: TextLabelMappingConfig,
    source_selection_config: TextSourceSelectionConfig,
    *,
    source_fingerprint: str,
    source_name: str,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_records: int | None = None,
    near_duplicate_limit: int = 1000,
    deduplicate_exact: bool = False,
    quarantine_conflicts: bool = True,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    validate_source_selection(source_selection_config)
    source_for_processing = source_df.head(max_records).copy() if max_records is not None else source_df.copy()
    before_distribution = {str(k).lower(): int(v) for k, v in source_for_processing[LABEL_COLUMN].value_counts().sort_index().items()}
    empty_summary = detect_empty_text(source_for_processing, TEXT_COLUMN)
    canonical_all = build_canonical_text_table(
        source_for_processing,
        label_mapping_config,
        source_fingerprint=source_fingerprint,
        source_name=source_name,
        max_records=None,
    )
    duplicate_groups = exact_duplicate_groups(canonical_all)
    near_candidates = bounded_near_duplicate_candidates(canonical_all, max_records=near_duplicate_limit)
    canonical_df, quarantine_df = apply_duplicate_policy(
        canonical_all,
        duplicate_groups,
        deduplicate_exact=deduplicate_exact,
        quarantine_conflicts=quarantine_conflicts,
    )
    after_distribution = {str(k): int(v) for k, v in canonical_df["canonical_label"].value_counts().sort_index().items()}
    leakage_checks = {
        "target_leakage_columns": detect_target_leakage_columns(source_df.columns, LABEL_COLUMN),
        "engineered_feature_leakage_columns": detect_engineered_feature_leakage(source_df.columns),
        "privacy_pattern_leakage_after_normalization": detect_privacy_pattern_leakage(canonical_df["normalized_text"].tolist()),
        "future_feature_contracts": [item.dict() for item in default_future_feature_contracts()],
    }
    overlap = {"exact_overlap_count": 0}
    if reference_df is not None:
        reference_canonical = canonicalize_text_dataframe(
            reference_df,
            label_mapping_config,
            source_fingerprint=source_fingerprint,
            source_name="reference-test",
            text_column=TEXT_COLUMN,
            label_column=LABEL_COLUMN,
        )
        overlap = validate_predefined_test_overlap(canonical_all, reference_canonical)
        leakage_checks["reference_test_overlap_count"] = overlap["exact_overlap_count"]
    report = TextPreprocessingReport(
        preprocessing_version=TEXT_PREPROCESSING_VERSION,
        feature_schema_version=TEXT_FEATURE_SCHEMA_VERSION,
        label_mapping_version=TEXT_LABEL_MAPPING_VERSION,
        privacy_ruleset_version=TEXT_PRIVACY_RULESET_VERSION,
        source_fingerprint=source_fingerprint,
        source_record_count=int(len(source_for_processing)),
        output_record_count=int(len(canonical_df)),
        excluded_record_count=int(len(canonical_all) - len(canonical_df)),
        empty_text_count=empty_summary["empty_text_count"],
        missing_text_count=empty_summary["missing_text_count"],
        exact_duplicate_group_count=int(len(duplicate_groups)),
        conflicting_duplicate_group_count=int(sum(1 for group in duplicate_groups if group.conflict)),
        near_duplicate_candidate_count=int(len(near_candidates)),
        label_distribution_before=before_distribution,
        label_distribution_after=after_distribution,
        privacy_replacement_summary=_privacy_totals(canonical_all),
        language_summary=_language_summary(canonical_all["normalized_text"].astype(str).tolist()[:5000]),
        length_summary=_length_summary(canonical_all),
        leakage_checks=leakage_checks,
        warnings=[
            "Research-only output; text alone cannot diagnose suicide risk or mental-health conditions.",
            "Privacy replacement is conservative and incomplete; it is not a guarantee of anonymization.",
            "No train/validation/test split, TF-IDF fitting, tokenizer fitting, model download, or model training was performed.",
            "No production chat records, PostgreSQL data, API routes, alerts, treatment recommendations, or SafeTalk logic were changed.",
            "Social-media/forum text may not represent Sri Lankan undergraduates or private support conversations.",
            "Duplicate and near-duplicate text may inflate later model performance if not handled during splitting.",
            "No user/group identifier is available for group-aware splitting.",
        ],
    )
    feature_schema = build_text_feature_schema(dataset_name=DATASET_NAME, dataset_version=DATASET_VERSION)
    outputs: dict[str, str] = {}
    if not validate_only:
        resolved_output_dir = output_dir.resolve(strict=False)
        paths.assert_not_raw_dataset_path(resolved_output_dir)
        if not paths.is_path_inside(paths.get_generated_root(), resolved_output_dir):
            raise ValueError("Text preprocessing outputs must be under generated/")
        safe_canonical = canonical_df[[column for column in CANONICAL_COLUMNS if column in canonical_df.columns]].copy()
        quarantine_safe = quarantine_df[["record_id", "text_hash", "canonical_label", "source_name"]].copy()
        outputs = {
            "canonical_csv": str(_write_csv(safe_canonical, resolved_output_dir / "canonical_text.csv", overwrite=overwrite)),
            "feature_schema_json": str(_write_json(feature_schema.to_safe_dict(), resolved_output_dir / "text_feature_schema.json", overwrite=overwrite)),
            "report_json": str(_write_json(report.to_safe_dict(), resolved_output_dir / "text_preprocessing_report.json", overwrite=overwrite)),
            "record_manifest_json": str(_write_json(_record_manifest(canonical_df, source_fingerprint), resolved_output_dir / "text_record_manifest.json", overwrite=overwrite)),
            "duplicate_manifest_json": str(_write_json(_duplicate_manifest(duplicate_groups, near_candidates), resolved_output_dir / "text_duplicate_manifest.json", overwrite=overwrite)),
            "conflict_quarantine_csv": str(_write_csv(quarantine_safe, resolved_output_dir / "text_conflict_quarantine.csv", overwrite=overwrite)),
            "label_distribution_json": str(
                _write_json({"before": before_distribution, "after": after_distribution}, resolved_output_dir / "text_label_distribution.json", overwrite=overwrite)
            ),
            "source_overlap_report_json": str(_write_json(_source_overlap_report(overlap), resolved_output_dir / "text_source_overlap_report.json", overwrite=overwrite)),
        }
        md_path = resolved_output_dir / "text_preprocessing_report.md"
        if md_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {md_path}")
        md_path.write_text(create_text_preprocessing_markdown(report), encoding="utf-8")
        outputs["report_markdown"] = str(md_path)

    return {
        "valid": True,
        "validate_only": validate_only,
        "source_rows": report.source_record_count,
        "output_rows": report.output_record_count,
        "excluded_rows": report.excluded_record_count,
        "exact_duplicate_groups": report.exact_duplicate_group_count,
        "conflicting_duplicate_groups": report.conflicting_duplicate_group_count,
        "near_duplicate_candidates": report.near_duplicate_candidate_count,
        "privacy_replacement_summary": report.privacy_replacement_summary,
        "label_distribution_before": before_distribution,
        "label_distribution_after": after_distribution,
        "report": report,
        "feature_schema": feature_schema,
        "outputs": outputs,
    }


def preprocess_text_dataset(
    dataset_config: DatasetConfig,
    preprocessing_config: PreprocessingConfig,
    label_mapping_config: TextLabelMappingConfig,
    source_selection_config: TextSourceSelectionConfig,
    fingerprint: DatasetFingerprint,
    *,
    output_dir: Path,
    overwrite: bool = False,
    validate_only: bool = False,
    max_records: int | None = None,
    near_duplicate_limit: int = 1000,
    deduplicate_exact: bool = False,
    quarantine_conflicts: bool = True,
    reference_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    source_df = load_text_source(dataset_config)
    return preprocess_text_dataframe(
        source_df,
        preprocessing_config,
        label_mapping_config,
        source_selection_config,
        source_fingerprint=fingerprint.combined_sha256,
        source_name=Path(str(dataset_config.source_path)).name,
        output_dir=output_dir,
        overwrite=overwrite,
        validate_only=validate_only,
        max_records=max_records,
        near_duplicate_limit=near_duplicate_limit,
        deduplicate_exact=deduplicate_exact,
        quarantine_conflicts=quarantine_conflicts,
        reference_df=reference_df,
    )


def synthetic_text_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Unique_ID": 1, "text": "I am not okay. email me at test@example.com", "status": "Depression"},
            {"Unique_ID": 2, "text": "I am not okay. email me at other@example.com", "status": "Depression"},
            {"Unique_ID": 3, "text": "Please see https://example.com and @helper", "status": "Anxiety"},
            {"Unique_ID": 4, "text": "Normal day with friends :)", "status": "Normal"},
            {"Unique_ID": 5, "text": "Normal day with friends :)", "status": "Suicidal"},
        ]
    )


def synthetic_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()

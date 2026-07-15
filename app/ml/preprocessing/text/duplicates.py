"""Duplicate and bounded near-duplicate analysis for text preprocessing."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
from typing import Any

import pandas as pd

from app.ml.preprocessing.text.schemas import DuplicateType, TextDuplicateGroup


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def exact_duplicate_groups(df: pd.DataFrame) -> list[TextDuplicateGroup]:
    required = {"record_id", "text_hash", "canonical_label", "source_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Cannot detect duplicates; missing columns: {sorted(missing)}")
    groups: list[TextDuplicateGroup] = []
    for text_hash, group in df.groupby("text_hash", sort=True):
        if len(group) < 2:
            continue
        labels = sorted(group["canonical_label"].astype(str).unique().tolist())
        groups.append(
            TextDuplicateGroup(
                duplicate_hash=str(text_hash),
                record_ids=sorted(group["record_id"].astype(str).tolist()),
                labels=labels,
                source_files=sorted(group["source_name"].astype(str).unique().tolist()),
                conflict=len(labels) > 1,
                duplicate_type=DuplicateType.EXACT,
            )
        )
    return groups


def apply_duplicate_policy(
    df: pd.DataFrame,
    duplicate_groups: list[TextDuplicateGroup],
    *,
    deduplicate_exact: bool = False,
    quarantine_conflicts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    conflict_ids = {record_id for group in duplicate_groups if group.conflict for record_id in group.record_ids}
    canonical = df.copy()
    quarantine = canonical[canonical["record_id"].isin(conflict_ids)].copy() if quarantine_conflicts else canonical.iloc[0:0].copy()
    if quarantine_conflicts and conflict_ids:
        canonical = canonical[~canonical["record_id"].isin(conflict_ids)].copy()
    if deduplicate_exact:
        canonical = canonical.sort_values(["text_hash", "record_id"], kind="mergesort").drop_duplicates("text_hash", keep="first")
    return canonical.reset_index(drop=True), quarantine.reset_index(drop=True)


def _word_shingles(text: str, size: int = 5) -> set[str]:
    words = text.split()
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + size]) for i in range(len(words) - size + 1)}


def bounded_near_duplicate_candidates(
    df: pd.DataFrame,
    *,
    max_records: int = 1000,
    threshold: float = 0.9,
    shingle_size: int = 5,
) -> list[dict[str, Any]]:
    if max_records <= 1 or df.empty:
        return []
    sample = df.head(max_records).copy()
    buckets: dict[str, list[tuple[str, str, set[str]]]] = defaultdict(list)
    for _, row in sample.iterrows():
        text = str(row.get("comparison_text", ""))
        shingles = _word_shingles(text, shingle_size)
        if not shingles:
            continue
        signature = min(shingles)
        buckets[signature].append((str(row["record_id"]), str(row["text_hash"]), shingles))

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for left, right in combinations(bucket, 2):
            left_id, left_hash, left_shingles = left
            right_id, right_hash, right_shingles = right
            pair = tuple(sorted((left_id, right_id)))
            if pair in seen or left_hash == right_hash:
                continue
            union = left_shingles | right_shingles
            if not union:
                continue
            score = len(left_shingles & right_shingles) / len(union)
            if score >= threshold:
                seen.add(pair)
                candidates.append({"record_ids": list(pair), "similarity": round(float(score), 6), "method": "word_shingle_jaccard_candidate"})
    return sorted(candidates, key=lambda item: (item["record_ids"], item["similarity"]))

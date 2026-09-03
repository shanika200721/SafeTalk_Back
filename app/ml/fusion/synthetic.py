"""Synthetic participant-aligned observations for offline fusion research.

This module intentionally does not import production application routes,
model registries, alerting, SafeTalk, database sessions, or counselor logic.
It creates research-only synthetic observations for fusion feasibility.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "synthetic_risk_label"
LATENT_COLUMN = "synthetic_latent_score"
IDENTIFIER_COLUMNS = {
    "synthetic_participant_id",
    "synthetic_observation_id",
    "observation_index",
    "observation_timestamp",
    "split",
}
FEATURE_COLUMNS = [
    "profile_score",
    "dass21_score",
    "mood_score",
    "text_score",
    "speech_score",
    "face_score",
    "behavioral_score",
]


def load_fusion_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON fusion configuration."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assign_label(score: float, thresholds: list[float], labels: list[str]) -> str:
    if score < thresholds[0]:
        return labels[0]
    if score < thresholds[1]:
        return labels[1]
    if score < thresholds[2]:
        return labels[2]
    return labels[3]


def _participant_splits(
    participant_ids: np.ndarray,
    split_proportions: dict[str, float],
    rng: np.random.Generator,
) -> dict[str, str]:
    shuffled = participant_ids.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(round(n * split_proportions["train"]))
    validation_end = train_end + int(round(n * split_proportions["validation"]))
    split_map: dict[str, str] = {}
    for pid in shuffled[:train_end]:
        split_map[str(pid)] = "train"
    for pid in shuffled[train_end:validation_end]:
        split_map[str(pid)] = "validation"
    for pid in shuffled[validation_end:]:
        split_map[str(pid)] = "test"
    return split_map


def generate_synthetic_fusion_dataset(
    config: dict[str, Any],
    seed: int | None = None,
    n_participants: int | None = None,
    shifted: bool = False,
) -> pd.DataFrame:
    """Generate participant-aligned synthetic observations.

    The synthetic risk category is generated from a hidden latent distress
    process plus target noise, not from any observed modality score.
    """

    seed_value = int(config["random_seed"] if seed is None else seed)
    participant_count = int(config["n_participants"] if n_participants is None else n_participants)
    rng = np.random.default_rng(seed_value)

    labels = list(config["risk_labels"])
    thresholds = list(config["target_generation"]["thresholds"])
    label_noise_sd = float(config["target_generation"]["label_noise_sd"])
    latent_cfg = config["latent_distress"]
    mixture_means = np.asarray(latent_cfg["mixture_means"], dtype=float)
    mixture_sds = np.asarray(latent_cfg["mixture_sds"], dtype=float)
    mixture_probs = np.asarray(latent_cfg["mixture_probabilities"], dtype=float)
    mixture_probs = mixture_probs / mixture_probs.sum()

    obs_cfg = config["observations_per_participant"]
    min_obs = int(obs_cfg["min"])
    max_obs = int(obs_cfg["max"])
    split_map = _participant_splits(
        np.asarray([f"syn-p{idx:05d}" for idx in range(participant_count)]),
        config["split_proportions"],
        rng,
    )

    rows: list[dict[str, Any]] = []
    observation_counter = 0
    for participant_index in range(participant_count):
        participant_id = f"syn-p{participant_index:05d}"
        component = rng.choice(len(mixture_means), p=mixture_probs)
        participant_latent = rng.normal(mixture_means[component], mixture_sds[component])
        participant_latent += rng.normal(0, latent_cfg["participant_random_effect_sd"])
        n_obs = int(rng.integers(min_obs, max_obs + 1))
        trend = rng.normal(0, latent_cfg["temporal_trend_sd"])

        for observation_index in range(n_obs):
            observation_counter += 1
            shared_noise = rng.normal(0, latent_cfg["shared_modality_noise_sd"])
            latent = participant_latent + trend * observation_index + rng.normal(0, latent_cfg["observation_noise_sd"])
            if shifted:
                shift_cfg = config["distribution_shift"]
                latent += float(shift_cfg["latent_mean_shift"])
            latent = float(np.clip(latent, 0, 100))
            target_score = float(np.clip(latent + rng.normal(0, label_noise_sd), 0, 100))

            row: dict[str, Any] = {
                "synthetic_participant_id": participant_id,
                "synthetic_observation_id": f"syn-o{observation_counter:07d}",
                "observation_index": observation_index,
                "observation_timestamp": f"2026-01-{(observation_index % 28) + 1:02d}T00:00:00Z",
                "synthetic_latent_score": round(target_score, 6),
                "synthetic_risk_label": _assign_label(target_score, thresholds, labels),
                "split": split_map[participant_id],
            }

            for feature in FEATURE_COLUMNS:
                modality = config["modality_generation"][feature]
                signal = float(modality["signal_strength"])
                noise_sd = float(modality["noise_sd"])
                bias = float(modality["bias"])
                shared_loading = float(modality["shared_noise_loading"])
                mean_anchor = float(modality["mean_anchor"])
                if shifted:
                    shift_cfg = config["distribution_shift"]["modality_adjustments"].get(feature, {})
                    bias += float(shift_cfg.get("bias_shift", 0.0))
                    noise_sd *= float(shift_cfg.get("noise_multiplier", 1.0))
                    signal *= float(shift_cfg.get("signal_multiplier", 1.0))
                value = signal * latent + (1.0 - signal) * mean_anchor + bias
                value += shared_loading * shared_noise + rng.normal(0, noise_sd)
                value = float(np.clip(value, 0, 100))
                if rng.random() < float(modality["missingness"]):
                    row[feature] = np.nan
                else:
                    row[feature] = round(value, 6)

            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sort_values(["synthetic_participant_id", "observation_index"]).reset_index(drop=True)


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    """Return predictor columns and exclude target-derived fields."""

    return [column for column in FEATURE_COLUMNS if column in df.columns]


def validate_synthetic_dataset(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Run deterministic integrity checks for synthetic fusion observations."""

    required = [
        "synthetic_participant_id",
        "synthetic_observation_id",
        "observation_index",
        "observation_timestamp",
        *FEATURE_COLUMNS,
        "synthetic_latent_score",
        "synthetic_risk_label",
        "split",
    ]
    findings: list[dict[str, Any]] = []
    missing_columns = [column for column in required if column not in df.columns]
    if missing_columns:
        findings.append({"check": "schema", "status": "failed", "missing_columns": missing_columns})

    if df["synthetic_observation_id"].duplicated().any():
        findings.append({"check": "duplicate_observation_ids", "status": "failed"})
    if df[["synthetic_participant_id", "observation_index"]].duplicated().any():
        findings.append({"check": "duplicate_participant_observation_index", "status": "failed"})

    split_participants = {
        split: set(df.loc[df["split"] == split, "synthetic_participant_id"])
        for split in ["train", "validation", "test"]
    }
    overlap = {
        "train_validation": sorted(split_participants["train"] & split_participants["validation"])[:5],
        "train_test": sorted(split_participants["train"] & split_participants["test"])[:5],
        "validation_test": sorted(split_participants["validation"] & split_participants["test"])[:5],
    }
    if any(overlap.values()):
        findings.append({"check": "participant_disjoint_splits", "status": "failed", "examples": overlap})

    split_observations = {
        split: set(df.loc[df["split"] == split, "synthetic_observation_id"])
        for split in ["train", "validation", "test"]
    }
    if (
        split_observations["train"] & split_observations["validation"]
        or split_observations["train"] & split_observations["test"]
        or split_observations["validation"] & split_observations["test"]
    ):
        findings.append({"check": "observation_overlap", "status": "failed"})

    for feature in FEATURE_COLUMNS:
        values = df[feature].dropna()
        if ((values < 0) | (values > 100)).any():
            findings.append({"check": "score_bounds", "status": "failed", "feature": feature})

    label_counts = df["synthetic_risk_label"].value_counts().to_dict()
    min_support = int(config["integrity_checks"]["minimum_class_support"])
    weak_classes = {label: int(count) for label, count in label_counts.items() if count < min_support}
    if weak_classes:
        findings.append({"check": "class_support", "status": "failed", "weak_classes": weak_classes})

    predictors = feature_columns_from_frame(df)
    leakage_columns = sorted((set(predictors) & {TARGET_COLUMN, LATENT_COLUMN}) | (set(predictors) & IDENTIFIER_COLUMNS))
    if leakage_columns:
        findings.append({"check": "target_leakage", "status": "failed", "columns": leakage_columns})

    label_codes = df[TARGET_COLUMN].map({label: idx for idx, label in enumerate(config["risk_labels"])})
    max_abs_corr = 0.0
    for feature in FEATURE_COLUMNS:
        valid = df[[feature]].join(label_codes.rename("label_code")).dropna()
        if len(valid) > 2:
            corr = abs(float(valid[feature].corr(valid["label_code"])))
            max_abs_corr = max(max_abs_corr, corr)
    if max_abs_corr >= float(config["integrity_checks"]["maximum_modality_label_correlation"]):
        findings.append({"check": "imperfect_modality_correlation", "status": "failed", "max_abs_correlation": max_abs_corr})

    deterministic = generate_synthetic_fusion_dataset(config).equals(generate_synthetic_fusion_dataset(config))
    if not deterministic:
        findings.append({"check": "deterministic_regeneration", "status": "failed"})

    return {
        "status": "passed" if not findings else "failed",
        "findings": findings,
        "row_count": int(len(df)),
        "participant_count": int(df["synthetic_participant_id"].nunique()),
        "class_counts": {str(key): int(value) for key, value in label_counts.items()},
        "feature_columns": predictors,
        "target_excluded_from_features": TARGET_COLUMN not in predictors and LATENT_COLUMN not in predictors,
        "max_abs_modality_label_correlation": max_abs_corr,
    }

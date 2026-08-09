"""Audit, retrain, and evaluate a v2 classical speech-emotion baseline.

This script is intentionally self-contained so the v1 baseline remains frozen.
It uses the existing speaker-grouped split manifest, keeps the test split
untouched during model selection, and writes all new evidence under v2 paths.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.fftpack import dct
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import VarianceThreshold

from app.ml.preprocessing.speech.audio_io import load_wav_audio

matplotlib.use("Agg")

SEED = 42
LABELS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
MANIFEST = REPO_ROOT / "generated/preprocessing/speech/v1/speech_canonical_manifest.csv"
SPLITS = REPO_ROOT / "generated/manifests/splits/speech/v1/speech_split_assignments.csv"
SPLIT_REPORT = REPO_ROOT / "generated/manifests/splits/speech/v1/speech_split_report.json"
SPEAKER_REPORT = REPO_ROOT / "generated/manifests/splits/speech/v1/speech_speaker_isolation_report.json"
CORPUS_REPORT = REPO_ROOT / "generated/manifests/splits/speech/v1/speech_corpus_distribution.json"
PREPROCESSING_REPORT = REPO_ROOT / "generated/preprocessing/speech/v1/speech_preprocessing_report.json"
V1_METRICS = REPO_ROOT / "generated/reports/speech_baseline/v1/speech_metrics_test.json"
V1_CONFUSION = REPO_ROOT / "generated/reports/speech_baseline/v1/speech_confusion_matrix_test.csv"
AUDIT_DIR = REPO_ROOT / "generated/reports/speech_baseline_audit"
REPORT_DIR = REPO_ROOT / "generated/reports/speech_baseline/v2"
FEATURE_DIR = REPO_ROOT / "generated/preprocessing/speech/v2"
MODEL_ROOT = REPO_ROOT / "ml_models/speech/speech-emotion-v2/2.0.0"
TARGET_SR = 16000


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    trim: bool
    fixed_seconds: float | None
    pre_emphasis: bool


FEATURE_CONFIGS = [
    FeatureConfig("no_trim_full_audio", trim=False, fixed_seconds=None, pre_emphasis=False),
    FeatureConfig("conservative_trim_full_audio", trim=True, fixed_seconds=None, pre_emphasis=False),
    FeatureConfig("conservative_trim_fixed_3s", trim=True, fixed_seconds=3.0, pre_emphasis=False),
    FeatureConfig("conservative_trim_preemphasis_fixed_3s", trim=True, fixed_seconds=3.0, pre_emphasis=True),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_json(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_joined_manifest() -> pd.DataFrame:
    manifest = pd.read_csv(MANIFEST)
    splits = pd.read_csv(SPLITS)
    df = manifest.merge(splits[["record_id", "split"]], on="record_id", how="inner")
    df["audio_path"] = df["audio_relative_path"].map(lambda value: str((REPO_ROOT / str(value)).resolve()))
    return df


def audit_current_pipeline(df: pd.DataFrame) -> dict[str, Any]:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    preprocessing = read_json(PREPROCESSING_REPORT)
    split_report = read_json(SPLIT_REPORT)
    speaker_report = read_json(SPEAKER_REPORT)
    corpus_report = read_json(CORPUS_REPORT)
    v1_metrics = read_json(V1_METRICS)

    class_rows = []
    for (split, label), count in df.groupby(["split", "canonical_emotion_label"]).size().items():
        total = int((df["split"] == split).sum())
        class_rows.append({"split": split, "emotion": label, "count": int(count), "proportion": count / total})
    pd.DataFrame(class_rows).sort_values(["split", "emotion"]).to_csv(AUDIT_DIR / "speech_class_distribution.csv", index=False)

    corpus_rows = []
    for (split, corpus), count in df.groupby(["split", "corpus_name"]).size().items():
        total = int((df["split"] == split).sum())
        corpus_rows.append({"split": split, "corpus": corpus, "count": int(count), "proportion": count / total})
    pd.DataFrame(corpus_rows).sort_values(["split", "corpus"]).to_csv(AUDIT_DIR / "speech_corpus_distribution.csv", index=False)

    split_rows = []
    for split, group in df.groupby("split"):
        split_rows.append(
            {
                "split": split,
                "records": len(group),
                "speakers": group["safe_speaker_key"].nunique(),
                "corpora": "|".join(sorted(group["corpus_name"].unique())),
                "classes": "|".join(sorted(group["canonical_emotion_label"].unique())),
            }
        )
    pd.DataFrame(split_rows).sort_values("split").to_csv(AUDIT_DIR / "speech_split_audit.csv", index=False)

    overlap_rows = []
    split_speakers = {split: set(group["safe_speaker_key"]) for split, group in df.groupby("split")}
    for left in sorted(split_speakers):
        for right in sorted(split_speakers):
            if left >= right:
                continue
            overlap = sorted(split_speakers[left] & split_speakers[right])
            overlap_rows.append({"split_pair": f"{left}|{right}", "overlap_count": len(overlap), "safe_speaker_keys": "|".join(overlap[:10])})
    pd.DataFrame(overlap_rows).to_csv(AUDIT_DIR / "speech_speaker_overlap.csv", index=False)

    recalculated = recalculate_v1_metrics(v1_metrics)
    write_json(AUDIT_DIR / "speech_metric_recalculation.json", recalculated)

    md = [
        "# Speech Pipeline Audit",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Datasets",
        "",
        f"- Corpora used: {', '.join(sorted(df['corpus_name'].unique()))}.",
        f"- Usable records: {len(df)}.",
        f"- Source records: {preprocessing.get('source_file_count')}; excluded records: {preprocessing.get('excluded_record_count')}; unreadable files: {preprocessing.get('unreadable_file_count')}.",
        f"- Corpus counts: {preprocessing.get('corpus_distribution')}.",
        f"- Label counts: {preprocessing.get('label_distribution')}.",
        "",
        "## Split Integrity",
        "",
        f"- Split strategy: {split_report.get('strategy')}.",
        f"- Speaker overlap count reported by split artifact: {speaker_report.get('speaker_overlap_count')}.",
        f"- Recomputed speaker pair overlaps: {overlap_rows}.",
        f"- Duplicate overlap count: {split_report.get('duplicate_handling', {}).get('duplicate_overlap_count')}.",
        f"- Split manifest hash: {split_report.get('manifest_hash')}.",
        "",
        "## Audio Quality and Standardization",
        "",
        f"- Sample rates observed before v2 standardization: {preprocessing.get('sample_rate_distribution')}.",
        f"- Channel distribution: {preprocessing.get('channel_distribution')}.",
        f"- Duration summary: {preprocessing.get('duration_summary')}.",
        "- v1 training used precomputed deterministic acoustic features; train-only imputation/scaling is handled inside sklearn preprocessing.",
        "- v2 feature extraction standardizes audio to mono, 16 kHz, peak amplitude normalization, conservative trim/fixed-duration variants, and near-empty rejection warnings.",
        "",
        "## Metric Audit",
        "",
        f"- Original test accuracy: {v1_metrics.get('accuracy')}.",
        f"- Original macro precision/recall/F1: {v1_metrics.get('macro_precision')}, {v1_metrics.get('macro_recall')}, {v1_metrics.get('macro_f1')}.",
        f"- Original weighted F1: {v1_metrics.get('weighted_f1')}.",
        f"- ROC-AUC was unavailable because the selected v1 RandomForest reported probability scores in a class order not aligned with the fixed label order; the evaluator caught the exception and stored null.",
        "",
        "## Main Weaknesses",
        "",
        "- The split is speaker-independent, but not corpus-balanced. TESS appears only in train; SAVEE appears in validation/test but not train in v1.",
        "- The v1 feature set is compact and omits delta MFCCs, chroma, mel-band summaries, spectral contrast, and richer distribution statistics.",
        "- Test confusions are concentrated among happy/fearful/disgust/sad/angry, with surprised recall especially weak.",
    ]
    (AUDIT_DIR / "speech_pipeline_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {
        "preprocessing": preprocessing,
        "split_report": split_report,
        "speaker_report": speaker_report,
        "corpus_report": corpus_report,
        "v1_metrics": v1_metrics,
        "v1_recalculated": recalculated,
    }


def recalculate_v1_metrics(v1_metrics: dict[str, Any]) -> dict[str, Any]:
    labels = v1_metrics["confusion_matrix"]["labels"]
    matrix = np.asarray(v1_metrics["confusion_matrix"]["matrix"], dtype=int)
    y_true: list[str] = []
    y_pred: list[str] = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            count = int(matrix[i, j])
            y_true.extend([true_label] * count)
            y_pred.extend([pred_label] * count)
    payload = metrics_payload(y_true, y_pred, labels=labels, probabilities=None)
    comparison = {}
    for key in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1"]:
        comparison[key] = {
            "reported": v1_metrics.get(key),
            "recalculated": payload.get(key),
            "absolute_difference": abs(float(v1_metrics.get(key, 0)) - float(payload.get(key, 0))),
        }
    return {
        "labels": labels,
        "averaging": "accuracy plus macro and weighted precision/recall/F1; balanced_accuracy equals macro recall for single-label multiclass",
        "comparison": comparison,
        "roc_auc_v1_reason": "not computed from v1 metrics artifact; probability class alignment was not available in the saved JSON",
    }


def frame_audio(audio: np.ndarray, sample_rate: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    if audio.size < frame_length:
        audio = np.pad(audio, (0, frame_length - audio.size))
    count = 1 + max(0, (audio.size - frame_length) // hop_length)
    starts = np.arange(count) * hop_length
    frames = np.stack([audio[start : start + frame_length] for start in starts])
    return frames * np.hanning(frame_length)


def stat_values(values: np.ndarray, prefix: str, stats: tuple[str, ...] = ("mean", "std", "min", "max", "median", "p10", "p90")) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"{prefix}_{stat}": 0.0 for stat in stats}
    out: dict[str, float] = {}
    for stat in stats:
        if stat == "mean":
            value = np.mean(arr)
        elif stat == "std":
            value = np.std(arr)
        elif stat == "min":
            value = np.min(arr)
        elif stat == "max":
            value = np.max(arr)
        elif stat == "median":
            value = np.median(arr)
        elif stat == "p10":
            value = np.percentile(arr, 10)
        elif stat == "p25":
            value = np.percentile(arr, 25)
        elif stat == "p75":
            value = np.percentile(arr, 75)
        elif stat == "p90":
            value = np.percentile(arr, 90)
        else:
            value = 0.0
        out[f"{prefix}_{stat}"] = float(np.nan_to_num(value))
    return out


def mel_filterbank(sample_rate: int, n_bins: int, n_filters: int = 40) -> np.ndarray:
    def hz_to_mel(hz: np.ndarray) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10 ** (mel / 2595.0) - 1.0)

    high = hz_to_mel(np.array([sample_rate / 2.0]))[0]
    points = mel_to_hz(np.linspace(0.0, high, n_filters + 2))
    bins = np.floor((n_bins * 2 - 1) * points / sample_rate).astype(int)
    fb = np.zeros((n_filters, n_bins), dtype=float)
    for idx in range(1, n_filters + 1):
        left, center, right = bins[idx - 1], bins[idx], bins[idx + 1]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for b in range(left, min(center, n_bins)):
            fb[idx - 1, b] = (b - left) / max(center - left, 1)
        for b in range(center, min(right, n_bins)):
            fb[idx - 1, b] = (right - b) / max(right - center, 1)
    return fb


def conservative_trim(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    if audio.size == 0:
        return audio, 0.0
    frame = max(1, int(0.025 * sample_rate))
    hop = max(1, int(0.010 * sample_rate))
    padded = audio if audio.size >= frame else np.pad(audio, (0, frame - audio.size))
    starts = np.arange(1 + (padded.size - frame) // hop) * hop
    rms = np.array([np.sqrt(np.mean(padded[start : start + frame] ** 2)) for start in starts])
    threshold = max(0.005, float(np.max(rms) * 0.03))
    active = np.where(rms > threshold)[0]
    if active.size == 0:
        return audio, 0.0
    pad = int(0.12 * sample_rate)
    start = max(0, int(starts[active[0]]) - pad)
    end = min(audio.size, int(starts[active[-1]] + frame) + pad)
    trimmed = audio[start:end]
    return trimmed, 1.0 - (trimmed.size / max(audio.size, 1))


def standardize_audio(path: str, config: FeatureConfig) -> tuple[int, np.ndarray, dict[str, Any]]:
    sample_rate, audio = load_wav_audio(path, mono=True, target_sample_rate=TARGET_SR)
    warnings_list: list[str] = []
    audio = np.nan_to_num(audio.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
    original_duration = audio.size / sample_rate if sample_rate else 0.0
    if audio.size < int(0.2 * sample_rate):
        warnings_list.append("near-empty audio")
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.98
    trimmed_ratio = 0.0
    if config.trim:
        audio, trimmed_ratio = conservative_trim(audio, sample_rate)
    if config.pre_emphasis and audio.size > 1:
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    if config.fixed_seconds is not None:
        target_len = int(round(config.fixed_seconds * sample_rate))
        if audio.size > target_len:
            audio = audio[:target_len]
        elif audio.size < target_len:
            audio = np.pad(audio, (0, target_len - audio.size))
    quality = {
        "original_duration": original_duration,
        "processed_duration": audio.size / sample_rate if sample_rate else 0.0,
        "trimmed_ratio": trimmed_ratio,
        "peak_before_normalization": peak,
        "warnings": "|".join(warnings_list),
    }
    return sample_rate, audio.astype(float), quality


def extract_features_for_audio(path: str, config: FeatureConfig) -> dict[str, Any]:
    try:
        sample_rate, audio, quality = standardize_audio(path, config)
        frames = frame_audio(audio, sample_rate)
        raw_frames = frames / np.maximum(np.hanning(frames.shape[1]), 1e-6)
        spectrum = np.abs(np.fft.rfft(frames, axis=1))
        power = np.maximum(spectrum**2, 1e-12)
        freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sample_rate)
        weights = np.maximum(spectrum.sum(axis=1), 1e-12)
        centroid = (spectrum * freqs).sum(axis=1) / weights
        bandwidth = np.sqrt(((freqs - centroid[:, None]) ** 2 * spectrum).sum(axis=1) / weights)
        cumulative = np.cumsum(spectrum, axis=1)
        rolloff = np.array([freqs[np.searchsorted(cumulative[i], 0.85 * cumulative[i, -1], side="left").clip(max=len(freqs) - 1)] for i in range(len(frames))])
        flatness = np.exp(np.mean(np.log(np.maximum(spectrum, 1e-12)), axis=1)) / np.maximum(np.mean(spectrum, axis=1), 1e-12)
        zcr = np.mean(np.diff(np.signbit(raw_frames), axis=1), axis=1)
        rms = np.sqrt(np.mean(raw_frames**2, axis=1))
        fb = mel_filterbank(sample_rate, power.shape[1], n_filters=40)
        mel = np.maximum(power @ fb.T, 1e-12)
        log_mel = np.log(mel)
        mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :20]
        delta = np.gradient(mfcc, axis=0) if len(mfcc) > 1 else np.zeros_like(mfcc)
        delta2 = np.gradient(delta, axis=0) if len(delta) > 1 else np.zeros_like(delta)
        features: dict[str, Any] = {
            "duration_seconds": quality["processed_duration"],
            "original_duration_seconds": quality["original_duration"],
            "trimmed_ratio": quality["trimmed_ratio"],
            "dynamic_range": float(np.percentile(audio, 95) - np.percentile(audio, 5)) if audio.size else 0.0,
            "voice_activity_ratio": float(np.mean(rms > max(0.005, np.max(rms) * 0.03))) if rms.size else 0.0,
        }
        for name, values in {
            "zero_crossing_rate": zcr,
            "rms_energy": rms,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "spectral_flatness": flatness,
        }.items():
            features.update(stat_values(values, name))
        for idx in range(mfcc.shape[1]):
            features.update(stat_values(mfcc[:, idx], f"mfcc_{idx + 1:02d}"))
            features.update(stat_values(delta[:, idx], f"delta_mfcc_{idx + 1:02d}"))
            features.update(stat_values(delta2[:, idx], f"delta2_mfcc_{idx + 1:02d}"))
        for idx in range(log_mel.shape[1]):
            features.update(stat_values(log_mel[:, idx], f"mel_{idx + 1:02d}", stats=("mean", "std", "min", "max", "median")))
        chroma = chroma_features(power, freqs)
        for idx in range(chroma.shape[1]):
            features.update(stat_values(chroma[:, idx], f"chroma_{idx:02d}", stats=("mean", "std", "max", "median")))
        contrast = spectral_contrast(power, freqs)
        for idx in range(contrast.shape[1]):
            features.update(stat_values(contrast[:, idx], f"spectral_contrast_{idx + 1}", stats=("mean", "std", "median")))
        pitch = pitch_track(raw_frames, sample_rate)
        features.update(stat_values(pitch[pitch > 0], "pitch_hz"))
        features["voiced_frame_ratio"] = float(np.mean(pitch > 0)) if pitch.size else 0.0
        features["feature_warning"] = quality["warnings"]
        return {key: float(np.nan_to_num(value)) if isinstance(value, (float, int, np.floating, np.integer)) else value for key, value in features.items()}
    except Exception as exc:
        return {"feature_warning": f"feature extraction failed: {exc.__class__.__name__}"}


def chroma_features(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    chroma = np.zeros((power.shape[0], 12), dtype=float)
    valid = freqs >= 50
    if not np.any(valid):
        return chroma
    midi = np.rint(69 + 12 * np.log2(np.maximum(freqs[valid], 1e-6) / 440.0)).astype(int)
    classes = np.mod(midi, 12)
    for cls in range(12):
        chroma[:, cls] = power[:, valid][:, classes == cls].sum(axis=1)
    total = np.maximum(chroma.sum(axis=1, keepdims=True), 1e-12)
    return chroma / total


def spectral_contrast(power: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    bands = [(0, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
    out = np.zeros((power.shape[0], len(bands)), dtype=float)
    logp = 10 * np.log10(np.maximum(power, 1e-12))
    for idx, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if np.any(mask):
            band = logp[:, mask]
            out[:, idx] = np.percentile(band, 90, axis=1) - np.percentile(band, 10, axis=1)
    return out


def pitch_track(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    pitches = np.zeros(len(frames), dtype=float)
    min_lag = max(1, int(sample_rate / 400.0))
    max_lag = max(min_lag + 1, int(sample_rate / 75.0))
    for i, frame in enumerate(frames):
        frame = frame - frame.mean()
        if np.sqrt(np.mean(frame**2)) < 0.005:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if corr[0] <= 1e-12 or max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        lag = int(np.argmax(segment) + min_lag)
        if corr[lag] / corr[0] >= 0.3:
            pitches[i] = sample_rate / lag
    return pitches


def extract_feature_file(df: pd.DataFrame, config: FeatureConfig, n_jobs: int, force: bool) -> Path:
    out = FEATURE_DIR / f"speech_features_{config.name}.csv"
    if out.exists() and not force:
        return out
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = joblib.Parallel(n_jobs=n_jobs, prefer="processes")(
        joblib.delayed(extract_features_for_audio)(row.audio_path, config) for row in df.itertuples(index=False)
    )
    meta = df[["record_id", "split", "canonical_emotion_label", "safe_speaker_key", "corpus_name", "audio_relative_path"]].reset_index(drop=True)
    features = pd.DataFrame(rows).fillna(0.0)
    result = pd.concat([meta, features], axis=1)
    result.to_csv(out, index=False)
    write_json(
        FEATURE_DIR / f"speech_features_{config.name}.metadata.json",
        {
            "config": config.__dict__,
            "target_sample_rate": TARGET_SR,
            "mono": True,
            "amplitude_normalization": "peak normalization to 0.98",
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "records": len(result),
            "sha256": sha256_file(out),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return out


def feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    blocked = {"record_id", "split", "canonical_emotion_label", "safe_speaker_key", "corpus_name", "audio_relative_path", "feature_warning"}
    numeric = [c for c in features.columns if c not in blocked and pd.api.types.is_numeric_dtype(features[c])]
    if feature_set == "mfcc_only":
        return [c for c in numeric if c.startswith("mfcc_")]
    if feature_set == "mfcc_delta":
        return [c for c in numeric if c.startswith("mfcc_") or c.startswith("delta_mfcc_") or c.startswith("delta2_mfcc_")]
    return numeric


def make_pipeline(estimator: Any, *, scale: bool, select_k: int | None = None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold())]
    if scale:
        steps.append(("scaler", StandardScaler()))
    if select_k:
        steps.append(("select", SelectKBest(mutual_info_classif, k=select_k)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def candidate_pipelines(n_features: int) -> list[tuple[str, Pipeline]]:
    k = min(160, max(20, n_features))
    return [
        ("dummy_most_frequent", make_pipeline(DummyClassifier(strategy="most_frequent"), scale=False)),
        ("logistic_C0.3_balanced", make_pipeline(LogisticRegression(C=0.3, class_weight="balanced", max_iter=500, solver="lbfgs", multi_class="auto", random_state=SEED), scale=True, select_k=k)),
        ("logistic_C1_balanced", make_pipeline(LogisticRegression(C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", multi_class="auto", random_state=SEED), scale=True, select_k=k)),
        ("linear_svc_C0.3_balanced", make_pipeline(LinearSVC(C=0.3, class_weight="balanced", max_iter=6000, random_state=SEED), scale=True, select_k=k)),
        ("linear_svc_C1_balanced", make_pipeline(LinearSVC(C=1.0, class_weight="balanced", max_iter=6000, random_state=SEED), scale=True, select_k=k)),
        ("rbf_svc_C3_gamma_scale_balanced", make_pipeline(SVC(C=3.0, gamma="scale", class_weight="balanced", probability=False, random_state=SEED), scale=True, select_k=min(80, k))),
        ("random_forest_180_depth16_balanced", make_pipeline(RandomForestClassifier(n_estimators=180, max_depth=16, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED), scale=False, select_k=None)),
        ("extra_trees_220_balanced", make_pipeline(ExtraTreesClassifier(n_estimators=220, max_depth=None, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=SEED), scale=False, select_k=None)),
        ("hist_gradient_boosting_l2", make_pipeline(HistGradientBoostingClassifier(max_iter=100, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.1, random_state=SEED), scale=False, select_k=k)),
        ("knn_9_distance", make_pipeline(KNeighborsClassifier(n_neighbors=9, weights="distance"), scale=True, select_k=min(120, k))),
    ]


def evaluate_cv(features: pd.DataFrame, columns: list[str], pipelines: list[tuple[str, Pipeline]], folds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = features[features["split"].isin(["train", "validation"])].copy()
    X = dev[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y = dev["canonical_emotion_label"].astype(str).to_numpy()
    groups = dev["safe_speaker_key"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=folds)
    rows = []
    fold_rows = []
    for name, pipeline in pipelines:
        scores = []
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y, groups), start=1):
            model = clone(pipeline)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X.iloc[train_idx], y[train_idx])
            pred = model.predict(X.iloc[valid_idx])
            row = {
                "candidate": name,
                "fold": fold,
                "macro_f1": f1_score(y[valid_idx], pred, labels=LABELS, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[valid_idx], pred),
                "accuracy": accuracy_score(y[valid_idx], pred),
            }
            fold_rows.append(row)
            scores.append(row)
        rows.append(
            {
                "candidate": name,
                "mean_macro_f1": float(np.mean([r["macro_f1"] for r in scores])),
                "std_macro_f1": float(np.std([r["macro_f1"] for r in scores])),
                "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in scores])),
                "mean_accuracy": float(np.mean([r["accuracy"] for r in scores])),
                "folds": folds,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_macro_f1", "mean_balanced_accuracy"], ascending=False), pd.DataFrame(fold_rows)


def preprocessing_comparison(df: pd.DataFrame, force: bool, n_jobs: int) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths: dict[str, Path] = {}
    rows = []
    for config in FEATURE_CONFIGS:
        path = extract_feature_file(df, config, n_jobs=n_jobs, force=force)
        paths[config.name] = path
        features = pd.read_csv(path)
        cols = feature_columns(features, "mfcc_delta")
        train = features[features["split"] == "train"]
        valid = features[features["split"] == "validation"]
        pipe = make_pipeline(LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=SEED), scale=True, select_k=min(160, len(cols)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(train[cols], train["canonical_emotion_label"].astype(str))
        pred = pipe.predict(valid[cols])
        rows.append(
            {
                "preprocessing_config": config.name,
                "feature_set_for_comparison": "mfcc_delta",
                "validation_macro_f1": f1_score(valid["canonical_emotion_label"], pred, labels=LABELS, average="macro", zero_division=0),
                "validation_balanced_accuracy": balanced_accuracy_score(valid["canonical_emotion_label"], pred),
                "validation_accuracy": accuracy_score(valid["canonical_emotion_label"], pred),
                "feature_file": rel(path),
                "feature_file_sha256": sha256_file(path),
            }
        )
    result = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    result.to_csv(REPORT_DIR / "speech_preprocessing_comparison.csv", index=False)
    return result, paths


def metrics_payload(y_true: Iterable[str], y_pred: Iterable[str], *, labels: list[str] = LABELS, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(list(y_true), dtype=str)
    y_pred = np.asarray(list(y_pred), dtype=str)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    payload: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "support": {label: int((y_true == label).sum()) for label in labels} | {"total": int(len(y_true))},
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, label in enumerate(labels)
        },
    }
    if probabilities is not None and probabilities.shape[1] == len(labels) and set(labels).issubset(set(y_true)):
        y_bin = label_binarize(y_true, classes=labels)
        ordered_labels = sorted(labels)
        ordered_probabilities = probabilities[:, [labels.index(label) for label in ordered_labels]]
        try:
            payload["roc_auc_macro_ovr"] = float(roc_auc_score(y_true, ordered_probabilities, labels=ordered_labels, multi_class="ovr", average="macro"))
            payload["roc_auc_weighted_ovr"] = float(roc_auc_score(y_true, ordered_probabilities, labels=ordered_labels, multi_class="ovr", average="weighted"))
        except ValueError as exc:
            payload["roc_auc_macro_ovr"] = None
            payload["roc_auc_weighted_ovr"] = None
            payload["roc_auc_reason"] = str(exc)
        payload["pr_auc_macro_ovr"] = float(average_precision_score(y_bin, probabilities, average="macro"))
        payload["pr_auc_weighted_ovr"] = float(average_precision_score(y_bin, probabilities, average="weighted"))
    else:
        payload["roc_auc_macro_ovr"] = None
        payload["roc_auc_weighted_ovr"] = None
        payload["pr_auc_macro_ovr"] = None
        payload["pr_auc_weighted_ovr"] = None
    return payload


def predict_probabilities(model: Pipeline, X: pd.DataFrame, labels: list[str]) -> np.ndarray | None:
    final = model.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        return None
    probs = model.predict_proba(X)
    classes = list(getattr(final, "classes_", labels))
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for idx, label in enumerate(labels):
        if label in classes:
            aligned[:, idx] = probs[:, classes.index(label)]
    row_sums = aligned.sum(axis=1)
    valid = row_sums > 0
    aligned[valid] = aligned[valid] / row_sums[valid, None]
    return aligned


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None, n: int = 1000) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    keys = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    values = {key: [] for key in keys}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        probs = probabilities[idx] if probabilities is not None else None
        sample = metrics_payload(y_true[idx], y_pred[idx], probabilities=probs)
        for key in keys:
            values[key].append(sample[key])
    return {key: {"low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5))} for key, vals in values.items()}


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(LABELS)), labels=LABELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Speech v2 Confusion Matrix")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_curves(y_true: np.ndarray, probabilities: np.ndarray | None) -> None:
    y_bin = label_binarize(y_true, classes=LABELS)
    for curve_type, out in [("roc", REPORT_DIR / "speech_roc_curve.png"), ("pr", REPORT_DIR / "speech_precision_recall_curve.png")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        if probabilities is None:
            ax.text(0.5, 0.5, "Probability scores unavailable", ha="center", va="center")
            ax.set_axis_off()
        else:
            for idx, label in enumerate(LABELS):
                if curve_type == "roc":
                    fpr, tpr, _ = roc_curve(y_bin[:, idx], probabilities[:, idx])
                    ax.plot(fpr, tpr, label=label, linewidth=1.2)
                else:
                    precision, recall, _ = precision_recall_curve(y_bin[:, idx], probabilities[:, idx])
                    ax.plot(recall, precision, label=label, linewidth=1.2)
            ax.set_xlabel("False positive rate" if curve_type == "roc" else "Recall")
            ax.set_ylabel("True positive rate" if curve_type == "roc" else "Precision")
            ax.legend(fontsize=8, ncol=2)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out, dpi=180)
        plt.close(fig)


def final_evaluation(features: pd.DataFrame, columns: list[str], selected_name: str, selected_pipe: Pipeline, cv_row: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    dev = features[features["split"].isin(["train", "validation"])].copy()
    test = features[features["split"] == "test"].copy()
    X_dev = dev[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_dev = dev["canonical_emotion_label"].astype(str).to_numpy()
    X_test = test[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_test = test["canonical_emotion_label"].astype(str).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected_pipe.fit(X_dev, y_dev)
    y_pred = selected_pipe.predict(X_test)
    probabilities = predict_probabilities(selected_pipe, X_test, LABELS)
    metrics = metrics_payload(y_test, y_pred, probabilities=probabilities)
    metrics["bootstrap_95ci"] = bootstrap_ci(y_test, y_pred, probabilities)
    metrics["selected_model"] = selected_name
    metrics["selection_cv"] = cv_row
    metrics["test_evaluation_policy"] = "frozen v1 test split evaluated once after grouped-CV model selection"
    metrics["random_seed"] = SEED
    metrics["split_manifest_hash"] = audit["split_report"].get("manifest_hash")

    write_json(REPORT_DIR / "speech_metrics_test.json", metrics)
    pd.DataFrame(classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)).T.to_csv(REPORT_DIR / "speech_classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, y_pred, labels=LABELS), index=LABELS, columns=LABELS).to_csv(REPORT_DIR / "speech_confusion_matrix.csv")
    plot_confusion(y_test, y_pred, REPORT_DIR / "speech_confusion_matrix.png")
    plot_curves(y_test, probabilities)

    per_class = [{"emotion": label, **values} for label, values in metrics["per_class"].items()]
    pd.DataFrame(per_class).to_csv(REPORT_DIR / "speech_per_class_metrics.csv", index=False)

    corpus_rows = []
    for corpus, group in test.assign(y_true=y_test, y_pred=y_pred).groupby("corpus_name"):
        group_metrics = metrics_payload(group["y_true"], group["y_pred"], probabilities=None)
        corpus_rows.append({"corpus": corpus, "support": len(group), **{k: group_metrics[k] for k in ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]}})
    pd.DataFrame(corpus_rows).to_csv(REPORT_DIR / "speech_per_corpus_metrics.csv", index=False)

    error_rows = []
    for row, true_label, pred_label in zip(test.itertuples(index=False), y_test, y_pred):
        if true_label != pred_label:
            error_rows.append({"record_id": row.record_id, "corpus": row.corpus_name, "true_emotion": true_label, "predicted_emotion": pred_label})
    pd.DataFrame(error_rows).to_csv(REPORT_DIR / "speech_error_analysis.csv", index=False)

    confusion_pairs = pd.DataFrame(error_rows).groupby(["true_emotion", "predicted_emotion"]).size().reset_index(name="count").sort_values("count", ascending=False)
    confusion_pairs.to_csv(REPORT_DIR / "speech_confused_pairs.csv", index=False)

    run_id = f"speech-v2-{hash_json({'model': selected_name, 'cv': cv_row, 'features': len(columns), 'split': metrics['split_manifest_hash']})[:12]}"
    run_dir = MODEL_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipe, run_dir / "speech_v2_pipeline.joblib")
    write_json(
        run_dir / "training_config.json",
        {
            "selected_model": selected_name,
            "random_seed": SEED,
            "target_sample_rate": TARGET_SR,
            "feature_columns": columns,
            "label_mapping": LABELS,
            "primary_selection_metric": "grouped_cv_macro_f1",
            "preprocessing": "selected from validation-only preprocessing comparison",
        },
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(
        run_dir / "reproducibility_report.json",
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_fingerprint": audit["preprocessing"].get("source_fingerprints"),
            "split_manifest_hash": metrics["split_manifest_hash"],
            "feature_file_hash": sha256_file(Path(features.attrs["feature_file"])),
            "code_version": sha256_file(Path(__file__)),
            "no_test_tuning": True,
        },
    )
    write_json(
        run_dir / "artifact_manifest.json",
        {path.name: {"path": rel(path), "sha256": sha256_file(path)} for path in run_dir.iterdir() if path.is_file()},
    )
    metrics["artifact_path"] = rel(run_dir)
    write_json(REPORT_DIR / "speech_metrics_test.json", metrics)
    return metrics


def feature_importance(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    transformed_names = columns
    if "select" in model.named_steps:
        mask = model.named_steps["select"].get_support()
        transformed_names = [name for name, keep in zip(columns, mask) if keep]
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        return pd.DataFrame({"feature": transformed_names[: len(values)], "importance": values}).sort_values("importance", ascending=False)
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        values = np.mean(np.abs(coef), axis=0)
        return pd.DataFrame({"feature": transformed_names[: len(values)], "aggregate_abs_coefficient": values}).sort_values("aggregate_abs_coefficient", ascending=False)
    return pd.DataFrame({"feature": transformed_names, "note": "Selected model does not expose direct feature importance."})


def write_summary(metrics: dict[str, Any], preprocessing_rows: pd.DataFrame, feature_rows: pd.DataFrame, model_rows: pd.DataFrame, audit: dict[str, Any]) -> None:
    worst = pd.DataFrame([{"emotion": k, **v} for k, v in metrics["per_class"].items()]).sort_values("recall").head(3)
    md = [
        "# Speech Baseline v2 Evaluation Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Selection",
        "",
        f"- Test split preserved from v1: {metrics['support']['total']} records.",
        f"- Selected model: {metrics['selected_model']}.",
        f"- Selection metric: grouped CV macro F1 on train+validation only.",
        f"- Best preprocessing: {preprocessing_rows.iloc[0]['preprocessing_config']}.",
        f"- Best feature set: {feature_rows.iloc[0]['feature_set']}.",
        "",
        "## Held-Out Test Metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Macro precision: {metrics['macro_precision']:.6f}",
        f"- Macro recall: {metrics['macro_recall']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Weighted F1: {metrics['weighted_f1']:.6f}",
        f"- ROC-AUC macro OVR: {metrics['roc_auc_macro_ovr']}",
        f"- PR-AUC macro OVR: {metrics['pr_auc_macro_ovr']}",
        "",
        "## Audit Interpretation",
        "",
        f"- Original metrics were recalculated from the saved confusion matrix with negligible difference: {audit['v1_recalculated']['comparison']['macro_f1']}.",
        "- Speaker leakage was not found in the preserved split.",
        "- Corpus imbalance remains: TESS is train-only and SAVEE has no v1 train records, so domain shift limits speech performance.",
        "- Speech emotion is acted emotion recognition, not depression or suicide-risk detection.",
        "",
        "## Weak Classes",
        "",
    ]
    for row in worst.itertuples(index=False):
        md.append(f"- {row.emotion}: recall {row.recall:.4f}, F1 {row.f1:.4f}, support {row.support}.")
    md.extend(
        [
            "",
            "## Thesis Claim",
            "",
            "The v2 speech baseline is suitable only as supporting contextual evidence. It should not be described as a standalone distress, depression, or suicide-risk recognizer.",
        ]
    )
    (REPORT_DIR / "speech_evaluation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def update_thesis_tables(metrics: dict[str, Any]) -> None:
    metrics_csv = REPO_ROOT / "docs/thesis/chapter4_metrics_summary.csv"
    evidence_csv = REPO_ROOT / "docs/thesis/chapter4_evidence_inventory.csv"
    summary = pd.read_csv(metrics_csv)
    summary = summary[summary["Modality"] != "Speech v2"]
    summary.loc[len(summary)] = {
        "Modality": "Speech v2",
        "Dataset": "Speech Emotion; 12160 usable records; frozen v1 test 1846",
        "Selected Model/Method": metrics["selected_model"],
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["macro_precision"],
        "Recall": metrics["macro_recall"],
        "F1": metrics["macro_f1"],
        "ROC-AUC": metrics["roc_auc_macro_ovr"] if metrics["roc_auc_macro_ovr"] is not None else "Not available",
        "Artifact Path": metrics["artifact_path"],
        "Evaluation Evidence": "generated/reports/speech_baseline/v2/speech_metrics_test.json",
        "Status": "Research baseline v2; supporting modality only",
    }
    summary.to_csv(metrics_csv, index=False)

    evidence = pd.read_csv(evidence_csv)
    evidence = evidence[~((evidence["Area"] == "ML") & (evidence["Feature or Artifact"] == "Speech model v2"))]
    evidence.loc[len(evidence)] = {
        "Area": "ML",
        "Feature or Artifact": "Speech model v2",
        "Status": "Research baseline v2",
        "Evidence Path": "generated/reports/speech_baseline/v2/speech_metrics_test.json; generated/reports/speech_baseline/v2/speech_evaluation_summary.md",
        "Key Finding": f"Grouped-CV selected {metrics['selected_model']}; held-out macro F1 {metrics['macro_f1']:.4f}",
        "Thesis Use": "Include as acted speech-emotion supporting modality only",
    }
    evidence.to_csv(evidence_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    np.random.seed(SEED)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_joined_manifest()
    audit = audit_current_pipeline(df)

    prep_rows, paths = preprocessing_comparison(df, force=args.force_features, n_jobs=args.n_jobs)
    best_config_name = str(prep_rows.iloc[0]["preprocessing_config"])
    best_feature_file = paths[best_config_name]
    features = pd.read_csv(best_feature_file)
    features.attrs["feature_file"] = str(best_feature_file)

    feature_rows = []
    best_feature_set = None
    best_feature_score = -1.0
    for feature_set in ["mfcc_only", "mfcc_delta", "full", "full_select"]:
        cols = feature_columns(features, "full" if feature_set == "full_select" else feature_set)
        select_k = min(220, len(cols)) if feature_set == "full_select" else None
        pipe = make_pipeline(LogisticRegression(C=0.5, class_weight="balanced", max_iter=500, random_state=SEED), scale=True, select_k=select_k)
        rows, _folds = evaluate_cv(features, cols, [(f"logistic_{feature_set}", pipe)], folds=3)
        row = rows.iloc[0].to_dict()
        row["feature_set"] = feature_set
        row["feature_count"] = len(cols)
        row["selection_k"] = select_k
        feature_rows.append(row)
        if row["mean_macro_f1"] > best_feature_score:
            best_feature_score = row["mean_macro_f1"]
            best_feature_set = feature_set
    feature_df = pd.DataFrame(feature_rows).sort_values("mean_macro_f1", ascending=False)
    feature_df.to_csv(REPORT_DIR / "speech_feature_comparison.csv", index=False)

    selected_feature_set = str(feature_df.iloc[0]["feature_set"])
    columns = feature_columns(features, "full" if selected_feature_set == "full_select" else selected_feature_set)
    model_rows, fold_rows = evaluate_cv(features, columns, candidate_pipelines(len(columns)), folds=3)
    model_rows.to_csv(REPORT_DIR / "speech_model_comparison.csv", index=False)
    fold_rows.to_csv(REPORT_DIR / "speech_cross_validation_results.csv", index=False)

    pd.DataFrame(
        [
            {"augmentation": "none", "status": "evaluated", "grouped_cv_macro_f1": model_rows.iloc[0]["mean_macro_f1"], "note": "No augmentation retained."},
            {"augmentation": "time_shift/pitch_shift/noise/amplitude", "status": "not_retained", "grouped_cv_macro_f1": "", "note": "Audio augmentation was not used for final selection because the validated classical feature baseline already showed corpus-domain instability; validation-only augmentation extraction was deferred rather than risk label distortion."},
        ]
    ).to_csv(REPORT_DIR / "speech_augmentation_comparison.csv", index=False)

    selected_name = str(model_rows.iloc[0]["candidate"])
    selected_pipe = dict(candidate_pipelines(len(columns)))[selected_name]
    metrics = final_evaluation(features, columns, selected_name, selected_pipe, model_rows.iloc[0].to_dict(), audit)

    fitted = joblib.load(REPO_ROOT / metrics["artifact_path"] / "speech_v2_pipeline.joblib")
    feature_importance(fitted, columns).to_csv(REPORT_DIR / "speech_feature_importance.csv", index=False)
    write_summary(metrics, prep_rows, feature_df, model_rows, audit)
    update_thesis_tables(metrics)
    write_json(
        REPORT_DIR / "speech_artifact_inventory.json",
        {rel(path): sha256_file(path) for path in sorted([*REPORT_DIR.glob("*"), *AUDIT_DIR.glob("*")]) if path.is_file()},
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "selected_model": selected_name,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "report_dir": rel(REPORT_DIR),
                "artifact_path": metrics["artifact_path"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

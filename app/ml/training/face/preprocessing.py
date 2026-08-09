"""Deterministic 48x48 grayscale preprocessing for classical Face baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler

from app.ml.common import paths
from app.ml.training.face.constants import FACE_FLATTENED_PIXEL_COUNT, FACE_IMAGE_SIZE
from app.ml.training.face.schemas import FaceImageBundle


def _resolve_image(relative_path: str) -> Path:
    candidate = Path(str(relative_path).replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError("image path must be repository-relative")
    return (paths.get_repository_root() / candidate).resolve(strict=False)


def load_48x48_grayscale(path: str | Path) -> np.ndarray:
    image_path = _resolve_image(str(path)) if not Path(path).is_absolute() else Path(path)
    with Image.open(image_path) as image:
        if image.size != FACE_IMAGE_SIZE:
            raise ValueError(f"face image dimensions must be 48x48, got {image.size}: {image_path}")
        gray = image.convert("L")
        array = np.asarray(gray, dtype=np.float32)
    if array.shape != FACE_IMAGE_SIZE:
        raise ValueError(f"face image array must be 48x48, got {array.shape}: {image_path}")
    return array / np.float32(255.0)


def image_statistics(array: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(array, bins=16, range=(0.0, 1.0), density=False)
    probs = hist.astype(np.float32) / np.float32(max(hist.sum(), 1))
    entropy = -float(np.sum([p * np.log2(p) for p in probs if p > 0]))
    gy, gx = np.gradient(array)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = float(np.mean(grad > 0.15))
    return np.asarray(
        [
            float(np.mean(array)),
            float(np.std(array)),
            float(np.percentile(array, 95) - np.percentile(array, 5)),
            edge_density,
            entropy,
        ],
        dtype=np.float32,
    )


def flatten_image(array: np.ndarray) -> np.ndarray:
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.shape[0] != FACE_FLATTENED_PIXEL_COUNT:
        raise ValueError("flattened face image must contain 2304 pixels")
    return flat


def load_face_images_for_rows(rows: pd.DataFrame, *, feature_set: str = "flattened_pixels") -> FaceImageBundle:
    X_rows: list[np.ndarray] = []
    y: list[str] = []
    for _, row in rows.sort_values("record_id").iterrows():
        array = load_48x48_grayscale(row["image_relative_path"])
        if feature_set == "flattened_pixels":
            X_rows.append(flatten_image(array))
        elif feature_set == "image_statistics":
            X_rows.append(image_statistics(array))
        else:
            raise ValueError(f"unsupported face feature set: {feature_set}")
        y.append(str(row["canonical_emotion_label"]))
    X = np.vstack(X_rows).astype(np.float32, copy=False) if X_rows else np.empty((0, 0), dtype=np.float32)
    if feature_set == "flattened_pixels":
        feature_names = [f"pixel_{index:04d}" for index in range(FACE_FLATTENED_PIXEL_COUNT)]
    else:
        feature_names = ["mean_intensity", "std_intensity", "contrast", "edge_density", "entropy"]
    return FaceImageBundle(X=X, y=y, rows=rows.sort_values("record_id").reset_index(drop=True), feature_names=feature_names)


def fit_train_only_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler(copy=False)
    scaler.fit(X_train)
    return scaler


def transform_with_scaler(scaler: StandardScaler | None, X: np.ndarray) -> np.ndarray:
    if scaler is None:
        return X.astype(np.float32, copy=False)
    transformed = scaler.transform(X.astype(np.float32, copy=False))
    return transformed.astype(np.float32, copy=False)


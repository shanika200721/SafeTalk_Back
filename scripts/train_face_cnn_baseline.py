from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

from app.ml.common import paths
from app.ml.training.face.constants import FACE_LABELS
from app.ml.training.face.data import build_face_training_bundle


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in df.columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_repo_path(relative_path: str) -> Path:
    candidate = Path(str(relative_path).replace("\\", "/"))
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError("face image paths must be repository-relative")
    return (paths.get_repository_root() / candidate).resolve(strict=False)


def load_image(path: str) -> np.ndarray:
    with Image.open(resolve_repo_path(path)) as image:
        gray = image.convert("L")
        if gray.size != (48, 48):
            raise ValueError(f"expected 48x48 face image, got {gray.size}: {path}")
        array = np.asarray(gray, dtype=np.float32) / np.float32(255.0)
    return array[..., np.newaxis]


def rows_to_arrays(rows: pd.DataFrame, label_to_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows.sort_values("record_id").reset_index(drop=True)
    images = np.stack([load_image(value) for value in ordered["image_relative_path"]]).astype(np.float32, copy=False)
    labels = np.asarray([label_to_index[str(value)] for value in ordered["canonical_emotion_label"]], dtype=np.int64)
    return images, labels


def build_cnn_model(seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = tf.keras.layers.RandomFlip("horizontal", seed=seed)(inputs)
    x = tf.keras.layers.RandomRotation(0.04, seed=seed)(x)
    x = tf.keras.layers.RandomZoom(0.08, seed=seed)(x)

    for filters, dropout in [(32, 0.10), (64, 0.15), (128, 0.20)]:
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Dropout(dropout, seed=seed)(x)

    x = tf.keras.layers.Conv2D(192, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(160, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35, seed=seed)(x)
    outputs = tf.keras.layers.Dense(len(FACE_LABELS), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet_model(seed: int) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    inputs = tf.keras.Input(shape=(48, 48, 1))
    x = tf.keras.layers.RandomFlip("horizontal", seed=seed)(inputs)
    x = tf.keras.layers.RandomRotation(0.04, seed=seed)(x)
    x = tf.keras.layers.Resizing(96, 96)(x)
    x = tf.keras.layers.Concatenate()([x, x, x])
    x = tf.keras.layers.Lambda(lambda value: tf.keras.applications.mobilenet_v2.preprocess_input(value * 255.0))(x)
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(96, 96, 3),
        pooling="avg",
    )
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.45, seed=seed)(x)
    outputs = tf.keras.layers.Dense(len(FACE_LABELS), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=8e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(seed: int, architecture: str) -> tf.keras.Model:
    if architecture == "cnn":
        return build_cnn_model(seed)
    if architecture == "mobilenet":
        return build_mobilenet_model(seed)
    raise ValueError(f"unsupported architecture: {architecture}")


def fine_tune_mobilenet(model: tf.keras.Model, trainable_layers: int = 24, learning_rate: float = 2e-5) -> None:
    base = next((layer for layer in model.layers if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")), None)
    if base is None:
        raise ValueError("MobileNetV2 base model not found")
    base.trainable = True
    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray, split: str) -> dict[str, Any]:
    labels = list(FACE_LABELS)
    y_pred = probabilities.argmax(axis=1)
    label_names_true = [labels[index] for index in y_true]
    label_names_pred = [labels[index] for index in y_pred]
    report = classification_report(label_names_true, label_names_pred, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(label_names_true, label_names_pred, labels=labels)
    y_bin = label_binarize(label_names_true, classes=labels)
    per_class = {}
    false_negatives = {}
    roc_auc_by_class = {}
    for index, label in enumerate(labels):
        false_negatives[label] = int(matrix[index, :].sum() - matrix[index, index])
        per_class[label] = {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
            "false_negatives": false_negatives[label],
        }
        fpr, tpr, _ = roc_curve(y_bin[:, index], probabilities[:, index])
        roc_auc_by_class[label] = float(auc(fpr, tpr))
    recalls = {label: per_class[label]["recall"] for label in labels}
    worst_class = min(labels, key=lambda label: (recalls[label], label))
    return {
        "split": split,
        "accuracy": float(accuracy_score(label_names_true, label_names_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(label_names_true, label_names_pred)),
        "macro_precision": float(precision_score(label_names_true, label_names_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(label_names_true, label_names_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(label_names_true, label_names_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(label_names_true, label_names_pred, average="weighted", zero_division=0)),
        "roc_auc_macro_ovr": float(roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")),
        "roc_auc_weighted_ovr": float(roc_auc_score(y_bin, probabilities, average="weighted", multi_class="ovr")),
        "roc_auc_by_class": roc_auc_by_class,
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": labels,
        "per_class": per_class,
        "false_negatives_by_class": false_negatives,
        "minimum_class_recall": float(recalls[worst_class]),
        "worst_performing_class": worst_class,
    }


def plot_confusion(metrics: dict[str, Any], path: Path) -> None:
    labels = metrics["labels"]
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted emotion")
    ax.set_ylabel("True emotion")
    ax.set_title("Figure 4.9. Confusion Matrix of the Facial Emotion Recognition Model")
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_roc(y_true: np.ndarray, probabilities: np.ndarray, metrics: dict[str, Any], path: Path) -> None:
    labels = list(FACE_LABELS)
    y_names = [labels[index] for index in y_true]
    y_bin = label_binarize(y_names, classes=labels)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    for index, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_bin[:, index], probabilities[:, index])
        class_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={class_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(
        "Figure 4.10. ROC Curve of the Facial Emotion Recognition Model\n"
        f"Macro AUC={metrics['roc_auc_macro_ovr']:.3f}, Weighted AUC={metrics['roc_auc_weighted_ovr']:.3f}"
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_tables(metrics: dict[str, Any], output_dir: Path, model_name: str, feature_set: str) -> None:
    summary_rows = [
        {"metric": "Model", "value": model_name},
        {"metric": "Feature set", "value": feature_set},
        {"metric": "Test observations", "value": sum(item["support"] for item in metrics["per_class"].values())},
        {"metric": "Accuracy", "value": f"{metrics['accuracy']:.4f}"},
        {"metric": "Balanced accuracy", "value": f"{metrics['balanced_accuracy']:.4f}"},
        {"metric": "Macro precision", "value": f"{metrics['macro_precision']:.4f}"},
        {"metric": "Macro recall", "value": f"{metrics['macro_recall']:.4f}"},
        {"metric": "Macro F1-score", "value": f"{metrics['macro_f1']:.4f}"},
        {"metric": "Weighted F1-score", "value": f"{metrics['weighted_f1']:.4f}"},
        {"metric": "ROC AUC macro OvR", "value": f"{metrics['roc_auc_macro_ovr']:.4f}"},
        {"metric": "ROC AUC weighted OvR", "value": f"{metrics['roc_auc_weighted_ovr']:.4f}"},
        {"metric": "Minimum class recall", "value": f"{metrics['minimum_class_recall']:.4f}"},
        {"metric": "Worst-performing class", "value": metrics["worst_performing_class"]},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "face_cnn_performance_table_test.csv", index=False)
    write_markdown_table(summary, output_dir / "face_cnn_performance_table_test.md")

    class_rows = []
    for label in FACE_LABELS:
        values = metrics["per_class"][label]
        class_rows.append(
            {
                "class": label,
                "precision": f"{values['precision']:.4f}",
                "recall": f"{values['recall']:.4f}",
                "f1-score": f"{values['f1']:.4f}",
                "support": int(values["support"]),
                "false negatives": int(values["false_negatives"]),
                "roc_auc_ovr": f"{metrics['roc_auc_by_class'][label]:.4f}",
            }
        )
    per_class = pd.DataFrame(class_rows)
    per_class.to_csv(output_dir / "face_cnn_per_class_metrics_test.csv", index=False)
    write_markdown_table(per_class, output_dir / "face_cnn_per_class_metrics_test.md")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train CNN face emotion baseline on leakage-safe face split.")
    parser.add_argument("--output-dir", default="generated/reports/face_cnn/v1")
    parser.add_argument("--model-dir", default="ml_models/face/face-emotion-cnn/1.0.0/face-cnn-v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--fine-tune-epochs", type=int, default=0)
    parser.add_argument("--architecture", choices=["cnn", "mobilenet"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=43107)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_face_training_bundle(
        deduplicated_manifest_path="generated/remediation/face/v1/face_deduplicated_manifest.csv",
        split_manifest_path="generated/manifests/splits/face/v2/face_split_manifest.json",
        split_assignments_path="generated/manifests/splits/face/v2/face_split_assignments.csv",
        quarantine_path="generated/remediation/face/v1/face_cross_label_quarantine.json",
        duplicate_decisions_path="generated/remediation/face/v1/face_remediation_decisions.csv",
        source_fingerprint_path="generated/manifests/fingerprints/face/facial-emotion-v1.json",
        max_train_records=args.max_train_records,
        require_replay=True,
    )
    label_to_index = {label: index for index, label in enumerate(FACE_LABELS)}
    X_train, y_train = rows_to_arrays(bundle.train, label_to_index)
    X_validation, y_validation = rows_to_arrays(bundle.validation, label_to_index)
    X_test, y_test = rows_to_arrays(bundle.test, label_to_index)

    classes = np.arange(len(FACE_LABELS), dtype=np.int64)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(index): float(weight) for index, weight in zip(classes, weights)}

    model = build_model(args.seed, args.architecture)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=args.patience,
            mode="max",
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, args.patience // 2),
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )
    history_payload = {f"initial_{key}": value for key, value in history.history.items()}
    if args.architecture == "mobilenet" and args.fine_tune_epochs > 0:
        fine_tune_mobilenet(model)
        fine_history = model.fit(
            X_train,
            y_train,
            validation_data=(X_validation, y_validation),
            epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=2,
        )
        history_payload.update({f"fine_tune_{key}": value for key, value in fine_history.history.items()})

    validation_probabilities = model.predict(X_validation, batch_size=args.batch_size, verbose=0)
    test_probabilities = model.predict(X_test, batch_size=args.batch_size, verbose=0)
    validation_metrics = evaluate_predictions(y_validation, validation_probabilities, "validation")
    test_metrics = evaluate_predictions(y_test, test_probabilities, "test")

    model.save(model_dir / "model.keras")
    write_json(model_dir / "training_config.json", vars(args))
    write_json(model_dir / "metrics.json", {"validation": validation_metrics, "test": test_metrics})
    write_json(output_dir / "face_cnn_metrics_validation.json", validation_metrics)
    write_json(output_dir / "face_cnn_metrics_test.json", test_metrics)
    write_json(output_dir / "face_cnn_training_history.json", history_payload)

    model_name = "face-emotion-mobilenetv2-transfer" if args.architecture == "mobilenet" else "face-emotion-cnn"
    feature_set = "48x48 grayscale pixels with ImageNet MobileNetV2 transfer" if args.architecture == "mobilenet" else "48x48 grayscale pixels"
    write_tables(test_metrics, output_dir, model_name, feature_set)
    plot_confusion(test_metrics, output_dir / "figure_4_9_face_confusion_matrix.png")
    plot_roc(y_test, test_probabilities, test_metrics, output_dir / "figure_4_10_face_roc_curve.png")

    print(
        json.dumps(
            {
                "status": "completed",
                "output_dir": str(output_dir),
                "model_dir": str(model_dir),
                "validation_accuracy": validation_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_roc_auc_macro_ovr": test_metrics["roc_auc_macro_ovr"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

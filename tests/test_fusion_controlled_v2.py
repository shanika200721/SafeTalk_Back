import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.fusion.weighted_late_fusion import (
    CLASS_ORDER,
    ThresholdConfig,
    WeightedLateFusion,
    classify_score,
    validate_weights,
    weighted_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "backend" / "scripts" / "run_controlled_fusion_v2.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
import run_controlled_fusion_v2 as fusion_v2


def test_weight_total_validation_accepts_selected_weights():
    validate_weights(fusion_v2.INITIAL_WEIGHTS)
    with pytest.raises(ValueError):
        validate_weights({"profile_score": 0.2, "text_score": 0.2})


def test_score_normalization_and_invalid_score_rejection():
    weights = {"profile_score": 0.5, "text_score": 0.5}
    assert weighted_score({"profile_score": 0.2, "text_score": 0.8}, weights) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        weighted_score({"profile_score": 1.2, "text_score": 0.8}, weights)


def test_generated_scores_are_normalized():
    df = fusion_v2.generate_controlled_dataset(seed=42, n=120)
    for modality in fusion_v2.MODALITIES:
        values = df[modality].dropna()
        assert values.between(0.0, 1.0).all()


def test_risk_threshold_boundary_handling():
    thresholds = ThresholdConfig(0.30, 0.50, 0.70)
    assert classify_score(0.2999, thresholds) == "Low"
    assert classify_score(0.30, thresholds) == "Moderate"
    assert classify_score(0.50, thresholds) == "High"
    assert classify_score(0.70, thresholds) == "Severe"


def test_missing_modality_weight_renormalization():
    weights = {"profile_score": 0.25, "text_score": 0.75}
    score = weighted_score({"profile_score": None, "text_score": 0.8}, weights)
    assert score == pytest.approx(0.8)


def test_reproducibility_with_seed_42_and_generation_consistency():
    first = fusion_v2.generate_controlled_dataset(seed=42, n=120)
    second = fusion_v2.generate_controlled_dataset(seed=42, n=120)
    pd.testing.assert_frame_equal(first, second)
    assert first["generation_seed"].eq(42).all()


def test_split_overlap_prevention():
    df = fusion_v2.generate_controlled_dataset(seed=42, n=240)
    train, validation, test, _ = fusion_v2.split_dataset(df)
    train_ids = set(train["synthetic_participant_id"])
    validation_ids = set(validation["synthetic_participant_id"])
    test_ids = set(test["synthetic_participant_id"])
    assert not train_ids & validation_ids
    assert not train_ids & test_ids
    assert not validation_ids & test_ids


def test_model_input_schema_validation():
    df = fusion_v2.generate_controlled_dataset(seed=42, n=40)
    features = fusion_v2.make_features(df)
    assert list(features.columns) == fusion_v2.FEATURE_COLUMNS
    for column in fusion_v2.AVAILABILITY_COLUMNS:
        assert set(features[column].dropna().unique()).issubset({0.0, 1.0})


def test_correct_class_order():
    assert fusion_v2.CLASS_ORDER == ["Low", "Moderate", "High", "Severe"]
    assert CLASS_ORDER == fusion_v2.CLASS_ORDER


def test_metrics_match_saved_confusion_matrix():
    metrics_path = REPO_ROOT / "generated" / "reports" / "fusion_controlled" / "v2" / "fusion_metrics_test.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = metrics["selected_learned_model_metrics"]
    matrix = np.asarray(selected["confusion_matrix"]["matrix"])
    total = matrix.sum()
    correct = np.trace(matrix)
    assert selected["confusion_matrix"]["labels"] == fusion_v2.CLASS_ORDER
    assert selected["accuracy"] == pytest.approx(correct / total)


def test_test_set_not_used_during_model_selection():
    metrics_path = REPO_ROOT / "generated" / "reports" / "fusion_controlled" / "v2" / "fusion_metrics_test.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["test_set_used_for_selection"] is False
    comparison = pd.read_csv(REPO_ROOT / "generated" / "reports" / "fusion_controlled" / "v2" / "fusion_model_comparison.csv")
    assert comparison["selection_metric"].eq("validation_macro_f1").all()


def test_weighted_model_predicts_with_expected_schema():
    model = WeightedLateFusion(fusion_v2.INITIAL_WEIGHTS, fusion_v2.INITIAL_THRESHOLDS)
    df = fusion_v2.generate_controlled_dataset(seed=42, n=20)
    predictions = model.predict(df[fusion_v2.MODALITIES])
    assert len(predictions) == len(df)
    assert set(predictions).issubset(set(fusion_v2.CLASS_ORDER))

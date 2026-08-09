import copy
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.fusion.evaluation import (
    DEFAULT_CONFIG,
    WeightedRuleBaseline,
    evaluate_predictions,
    fit_learned_model,
    model_candidates,
    run_missing_modality_robustness,
    sha256_file,
    write_model_card,
    write_table,
)
from app.ml.fusion.synthetic import (
    FEATURE_COLUMNS,
    LATENT_COLUMN,
    TARGET_COLUMN,
    generate_synthetic_fusion_dataset,
    load_fusion_config,
    validate_synthetic_dataset,
)


def small_config():
    config = copy.deepcopy(load_fusion_config(DEFAULT_CONFIG))
    config["n_participants"] = 140
    config["integrity_checks"]["minimum_class_support"] = 1
    config["robustness"]["seed_stability_seeds"] = [101, 102]
    config["robustness"]["sample_size_participants"] = [80, 120]
    return config


def test_synthetic_generation_is_deterministic_and_schema_valid():
    config = small_config()
    first = generate_synthetic_fusion_dataset(config)
    second = generate_synthetic_fusion_dataset(config)

    pd.testing.assert_frame_equal(first, second)
    required = {
        "synthetic_participant_id",
        "synthetic_observation_id",
        "observation_index",
        "observation_timestamp",
        *FEATURE_COLUMNS,
        LATENT_COLUMN,
        TARGET_COLUMN,
        "split",
    }
    assert required.issubset(first.columns)
    integrity = validate_synthetic_dataset(first, config)
    assert integrity["status"] == "passed"
    assert integrity["target_excluded_from_features"] is True


def test_score_bounds_and_grouped_splits_have_no_overlap():
    config = small_config()
    df = generate_synthetic_fusion_dataset(config)

    for feature in FEATURE_COLUMNS:
        values = df[feature].dropna()
        assert values.between(0, 100).all()

    participants_by_split = {
        split: set(df.loc[df["split"] == split, "synthetic_participant_id"])
        for split in ["train", "validation", "test"]
    }
    assert not (participants_by_split["train"] & participants_by_split["validation"])
    assert not (participants_by_split["train"] & participants_by_split["test"])
    assert not (participants_by_split["validation"] & participants_by_split["test"])
    assert not df["synthetic_observation_id"].duplicated().any()


def test_target_and_latent_columns_are_not_predictors():
    assert TARGET_COLUMN not in FEATURE_COLUMNS
    assert LATENT_COLUMN not in FEATURE_COLUMNS
    assert "synthetic_participant_id" not in FEATURE_COLUMNS
    assert "split" not in FEATURE_COLUMNS


def test_train_only_preprocessing_fit_for_learned_pipeline():
    config = small_config()
    df = generate_synthetic_fusion_dataset(config)
    logistic_candidates = model_candidates(config)["logistic_regression"][:1]
    result = fit_learned_model("logistic_regression", logistic_candidates, df, config["risk_labels"])

    imputer = result.estimator.named_steps["imputer"]
    train_medians = df.loc[df["split"] == "train", FEATURE_COLUMNS].median(numeric_only=True).to_numpy()
    full_medians = df[FEATURE_COLUMNS].median(numeric_only=True).to_numpy()
    np.testing.assert_allclose(imputer.statistics_, train_medians)
    assert not np.allclose(imputer.statistics_, full_medians)


def test_weighted_baseline_reproducibility_and_missing_values():
    config = small_config()
    df = generate_synthetic_fusion_dataset(config)
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"].copy()
    test.loc[test.index[:3], "speech_score"] = np.nan

    first = WeightedRuleBaseline(
        config["weighted_rule_baseline"]["weights"],
        config["weighted_rule_baseline"]["thresholds"],
        config["risk_labels"],
    ).fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    second = WeightedRuleBaseline(
        config["weighted_rule_baseline"]["weights"],
        config["weighted_rule_baseline"]["thresholds"],
        config["risk_labels"],
    ).fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])

    assert first.predict(test[FEATURE_COLUMNS]).tolist() == second.predict(test[FEATURE_COLUMNS]).tolist()


def test_metric_schema_contains_required_safety_fields():
    labels = ["low", "moderate", "high", "severe"]
    y_true = pd.Series(["low", "moderate", "high", "severe", "severe"])
    y_pred = np.asarray(["low", "high", "moderate", "severe", "high"])
    probabilities = np.asarray(
        [
            [0.8, 0.1, 0.05, 0.05],
            [0.1, 0.3, 0.5, 0.1],
            [0.1, 0.4, 0.3, 0.2],
            [0.05, 0.05, 0.2, 0.7],
            [0.05, 0.05, 0.6, 0.3],
        ]
    )

    metrics = evaluate_predictions(y_true, y_pred, labels, probabilities)
    for key in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "per_class", "confusion_matrix"]:
        assert key in metrics
    assert metrics["false_negatives_high_severe"] == {"high": 1, "severe": 1}
    assert metrics["calibration"]["available"] is True


def test_artifact_hashing_and_markdown_report_generation(tmp_path):
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("fusion feasibility\n", encoding="utf-8")
    assert sha256_file(artifact) == sha256_file(artifact)

    table = pd.DataFrame([{"model": "logistic_regression", "macro_f1": 0.5}])
    csv_path = tmp_path / "table.csv"
    md_path = tmp_path / "table.md"
    write_table(table, csv_path, md_path)
    assert csv_path.exists()
    assert "| model | macro_f1 |" in md_path.read_text(encoding="utf-8")


def test_missing_modality_robustness_is_reproducible():
    config = small_config()
    df = generate_synthetic_fusion_dataset(config)
    candidates = model_candidates(config)["gaussian_naive_bayes"][:1]
    result = fit_learned_model("gaussian_naive_bayes", candidates, df, config["risk_labels"])

    first = run_missing_modality_robustness(df, config, result.estimator, "gaussian_naive_bayes")
    second = run_missing_modality_robustness(df, config, result.estimator, "gaussian_naive_bayes")
    pd.testing.assert_frame_equal(first, second)


def test_model_card_disclaimers_and_production_isolation(tmp_path):
    config = small_config()
    df = generate_synthetic_fusion_dataset(config)
    candidates = model_candidates(config)["gaussian_naive_bayes"][:1]
    result = fit_learned_model("gaussian_naive_bayes", candidates, df, config["risk_labels"])
    card_path = tmp_path / "model_card.md"
    write_model_card(card_path, "gaussian_naive_bayes", {"gaussian_naive_bayes": result})
    card = card_path.read_text(encoding="utf-8")

    required = [
        "Offline research-only synthetic late-fusion feasibility analysis",
        "diagnosis",
        "autonomous intervention",
        "counselor alerting",
        "DASS21 is not a suicide-risk classifier",
        "Emotion is not suicide risk",
        "not a clinical diagnostic or autonomous suicide-prevention system",
    ]
    for phrase in required:
        assert phrase in card

    fusion_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (Path(__file__).resolve().parents[1] / "app" / "ml" / "fusion").glob("*.py")
    )
    prohibited_runtime_terms = [
        "from app.routes",
        "import app.routes",
        "risk_assessor =",
        "from app.services.model_registry",
        "import app.services.model_registry",
        "SessionLocal",
        "from app.ml.safetalk",
        "from app.ml.enhanced_safetalk",
        "from app.ml.counselor_safetalk",
    ]
    for term in prohibited_runtime_terms:
        assert term not in fusion_sources

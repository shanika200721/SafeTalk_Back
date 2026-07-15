from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ml.preprocessing.dass21.constants import (
    DASS21_EXPECTED_ITEMS,
    DASS21_FEATURE_SCHEMA_VERSION,
    DASS21_ITEM_MAPPING,
    DASS21_ITEM_MAPPING_VERSION,
    DASS21_SCORING_VERSION,
    DASS21_SEVERITY_THRESHOLDS,
    DASS42_TO_DASS21_SOURCE_COLUMNS,
    get_threshold_metadata,
)
from app.ml.preprocessing.dass21.dataset_mapping import (
    default_mapping_config,
    identify_demographic_columns,
    identify_response_columns,
    identify_timing_columns,
    inspect_dass_dataset_columns,
    map_dataset_row_to_dass21_responses,
    validate_dataset_item_mapping,
)
from app.ml.preprocessing.dass21.scoring import (
    apply_dass21_multiplier,
    calculate_subscale_raw_scores,
    classify_anxiety_severity,
    classify_depression_severity,
    classify_stress_severity,
    convert_database_record,
    convert_frontend_payload,
    explain_dass21_result,
    normalize_subscale_score,
    score_dass21,
    validate_dass21_responses,
)
from app.models.database_models import Alert, DASS21Assessment
from app.routes import assessments
from app.utils.dass21_calculator import DASS21Calculator


def full_response(value: int) -> dict[str, int]:
    return {item: value for item in DASS21_EXPECTED_ITEMS}


def full_dataset_columns() -> list[str]:
    columns = []
    for index in range(1, 43):
        columns.extend([f"Q{index}A", f"Q{index}I", f"Q{index}E"])
    columns.extend(["country", "source", "introelapse", "testelapse", "surveyelapse", "age", "gender", "major"])
    return columns


def test_authoritative_item_mapping_has_exactly_seven_unique_items_per_subscale():
    all_items = []
    for subscale, items in DASS21_ITEM_MAPPING.items():
        assert len(items) == 7, subscale
        all_items.extend(items)

    assert sorted(all_items, key=lambda item: int(item[1:])) == list(DASS21_EXPECTED_ITEMS)
    assert len(set(all_items)) == 21


def test_all_zero_responses_are_normal_and_deterministic():
    first = score_dass21(full_response(0))
    second = score_dass21(full_response(0))

    assert first.depression.raw_score == 0
    assert first.depression.multiplied_score == 0
    assert first.depression.severity == "normal"
    assert first.anxiety.severity == "normal"
    assert first.stress.severity == "normal"
    assert first.dict(exclude={"scored_at"}) == second.dict(exclude={"scored_at"})


def test_maximum_responses_score_to_extremely_severe():
    result = score_dass21(full_response(3))

    assert result.depression.raw_score == 21
    assert result.depression.multiplied_score == 42
    assert result.anxiety.multiplied_score == 42
    assert result.stress.multiplied_score == 42
    assert result.depression.severity == "extremely_severe"
    assert result.anxiety.severity == "extremely_severe"
    assert result.stress.severity == "extremely_severe"


def test_known_mixed_response_example_raw_totals_and_multiplier():
    responses = full_response(0)
    for item in DASS21_ITEM_MAPPING["depression"]:
        responses[item] = 1
    for item in DASS21_ITEM_MAPPING["anxiety"]:
        responses[item] = 2
    for item in DASS21_ITEM_MAPPING["stress"]:
        responses[item] = 3

    raw = calculate_subscale_raw_scores(responses)
    result = score_dass21(responses)

    assert raw == {"depression": 7, "anxiety": 14, "stress": 21}
    assert result.depression.multiplied_score == 14
    assert result.anxiety.multiplied_score == 28
    assert result.stress.multiplied_score == 42


def test_no_double_multiplication_for_already_multiplied_values():
    assert apply_dass21_multiplier(14, already_multiplied=True) == 14
    assert apply_dass21_multiplier(14, already_multiplied=False) == 28


@pytest.mark.parametrize(
    "classifier,bounds",
    [
        (classify_depression_severity, DASS21_SEVERITY_THRESHOLDS["depression"]),
        (classify_anxiety_severity, DASS21_SEVERITY_THRESHOLDS["anxiety"]),
        (classify_stress_severity, DASS21_SEVERITY_THRESHOLDS["stress"]),
    ],
)
def test_severity_boundaries(classifier, bounds):
    for severity, (minimum, maximum) in bounds.items():
        assert classifier(minimum) == severity
        assert classifier(maximum) == severity


def test_invalid_negative_response_response_above_max_missing_extra_and_duplicate_fail():
    with pytest.raises(ValueError, match="between 0 and 3"):
        validate_dass21_responses({**full_response(0), "Q1": -1})
    with pytest.raises(ValueError, match="between 0 and 3"):
        validate_dass21_responses({**full_response(0), "Q1": 4})
    with pytest.raises(ValueError, match="Missing"):
        validate_dass21_responses({"Q1": 0})
    with pytest.raises(ValueError, match="Unknown"):
        validate_dass21_responses({**full_response(0), "Q22": 0})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dass21_responses([("Q1", 0), ("Q1", 1)])


def test_incomplete_scoring_requires_explicit_policy_and_does_not_impute():
    with pytest.raises(ValueError, match="Incomplete"):
        score_dass21({"Q1": 0}, allow_incomplete=True)


def test_normalization_is_always_0_to_100():
    assert normalize_subscale_score(-2) == 0
    assert normalize_subscale_score(21) == 50
    assert normalize_subscale_score(100) == 100


def test_scored_at_must_be_timezone_aware_and_result_uses_timezone():
    with pytest.raises(ValueError, match="timezone-aware"):
        score_dass21(full_response(0), scored_at=datetime(2026, 7, 14))

    result = score_dass21(full_response(0), scored_at=datetime(2026, 7, 14, tzinfo=timezone.utc))
    assert result.scored_at.tzinfo is not None


def test_frontend_payload_conversion_and_legacy_wrapper_match_authoritative_module():
    responses = [0] * 21
    typed = convert_frontend_payload({"responses": responses})
    result = score_dass21(typed)
    legacy = DASS21Calculator.calculate(responses)

    assert legacy["depression_score"] == result.depression.multiplied_score
    assert legacy["anxiety_score"] == result.anxiety.multiplied_score
    assert legacy["stress_score"] == result.stress.multiplied_score
    assert {"depression_score", "anxiety_score", "stress_score", "total_dass21_score"} <= set(legacy)


def test_database_record_conversion_uses_responses_or_existing_totals_without_double_multiplication():
    with_responses = SimpleNamespace(responses=[1] * 21)
    scored = convert_database_record(with_responses)
    assert scored.depression.raw_score == 7
    assert scored.depression.multiplied_score == 14

    with_totals = SimpleNamespace(depression_score=14, anxiety_score=8, stress_score=20)
    converted = convert_database_record(with_totals)
    assert converted.depression.raw_score == 7
    assert converted.depression.multiplied_score == 14
    assert not converted.is_complete


def test_explanation_does_not_create_suicide_risk_alerts_or_recommendations():
    explanation = explain_dass21_result(score_dass21(full_response(0)))
    text = json.dumps(explanation)

    assert "risk_level" not in explanation
    assert "alert" not in explanation
    assert "treatment" not in explanation
    assert "raw answers" not in text.lower()


def test_threshold_metadata_versions_are_available():
    metadata = get_threshold_metadata()

    assert metadata["scoring_version"] == DASS21_SCORING_VERSION
    assert DASS21_FEATURE_SCHEMA_VERSION == "1.0.0"
    assert DASS21_ITEM_MAPPING_VERSION == "1.0.0"
    assert metadata["thresholds"]["depression"]["extremely_severe"]["minimum"] == 28


def test_dataset_column_identification_excludes_timing_demographics_and_metadata():
    columns = full_dataset_columns()
    inspection = inspect_dass_dataset_columns(columns)

    assert len(identify_response_columns(columns)) == 42
    assert len(identify_timing_columns(columns)) == 42
    assert "age" in identify_demographic_columns(columns)
    assert inspection["questionnaire_source"] == "DASS-42"
    assert "Q1E" not in inspection["response_columns"]


def test_valid_dataset_mapping_and_dass42_to_dass21_subset():
    result = validate_dataset_item_mapping(full_dataset_columns(), default_mapping_config())

    assert result["mapping_success"] is True
    assert result["response_column_count"] == 42
    assert result["excluded_timing_column_count"] == 42
    assert DASS42_TO_DASS21_SOURCE_COLUMNS["Q1"] == "Q22A"


def test_dataset_mapping_missing_response_duplicate_and_bad_subscale_counts_fail():
    columns = [column for column in full_dataset_columns() if column != "Q22A"]
    with pytest.raises(ValueError, match="Missing expected"):
        validate_dataset_item_mapping(columns, default_mapping_config())

    duplicate = default_mapping_config()
    duplicate["selected_dass21_items"] = [dict(item) for item in duplicate["selected_dass21_items"]]
    duplicate["selected_dass21_items"][1]["target_question_id"] = "Q1"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_dataset_item_mapping(full_dataset_columns(), duplicate)

    bad_counts = default_mapping_config()
    bad_counts["selected_dass21_items"] = [dict(item) for item in bad_counts["selected_dass21_items"]]
    bad_counts["selected_dass21_items"][0]["subscale"] = "depression"
    with pytest.raises(ValueError, match="7 mapped items"):
        validate_dataset_item_mapping(full_dataset_columns(), bad_counts)


def test_dataset_response_transformation_and_bounded_sample_scoring_without_writing_output_data():
    row = {column: "1" for column in full_dataset_columns()}
    responses = map_dataset_row_to_dass21_responses(row, default_mapping_config())
    result = score_dass21(responses)

    assert set(responses) == set(DASS21_EXPECTED_ITEMS)
    assert all(value == 0 for value in responses.values())
    assert result.is_complete


def test_response_transformation_rejects_invalid_source_values():
    row = {column: "1" for column in full_dataset_columns()}
    row["Q22A"] = "5"

    with pytest.raises(ValueError, match="between 1 and 4"):
        map_dataset_row_to_dass21_responses(row, default_mapping_config())


def test_privacy_safe_reports_contain_no_raw_answers_demographics_or_participant_ids():
    result = score_dass21(full_response(2))
    report = explain_dass21_result(result)
    report_text = json.dumps(report)

    assert "responses" not in report_text
    assert "student_id" not in report_text
    assert "participant" not in report_text
    assert "gender" not in report_text
    assert "major" not in report_text


class FakeDB:
    def __init__(self):
        self.added = []

    def add(self, item):
        self.added.append(item)

    def commit(self):
        pass

    def refresh(self, item):
        item.id = item.id or 1
        if item.created_at is None:
            item.created_at = datetime.now(timezone.utc)


def test_assessment_submission_endpoint_scoring_stored_values_and_no_alert_creation():
    app = FastAPI()
    fake_db = FakeDB()
    app.include_router(assessments.router)
    app.dependency_overrides[assessments.get_current_user] = lambda: SimpleNamespace(id=123)
    app.dependency_overrides[assessments.get_db] = lambda: fake_db

    client = TestClient(app)
    response = client.post("/api/assessments/dass21", json={"responses": [0] * 21})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["depression_score"] == 0
    assert payload["anxiety_score"] == 0
    assert payload["stress_score"] == 0
    assert payload["total_dass21_score"] == 0
    assert payload["depression_severity"] == "Normal"
    assert any(isinstance(item, DASS21Assessment) for item in fake_db.added)
    assert not any(isinstance(item, Alert) for item in fake_db.added)

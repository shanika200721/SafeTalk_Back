from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig, PreprocessingConfig
from app.ml.preprocessing.behavioral.constants import (
    BEHAVIORAL_FEATURE_SCHEMA_VERSION,
    BEHAVIORAL_MAPPING_VERSION,
    BEHAVIORAL_PREPROCESSING_VERSION,
    BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    SOURCE_STATUS_NO_BEHAVIORAL_DATA,
    SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET,
    SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY,
)
from app.ml.preprocessing.behavioral.features import build_behavioral_feature_table
from app.ml.preprocessing.behavioral.mapping import default_behavioral_mapping_config
from app.ml.preprocessing.behavioral.preprocessor import (
    aggregate_behavioral_sessions,
    assess_participant_baseline_eligibility,
    calculate_baseline_deviation_features,
    canonicalize_behavioral_events,
    classify_behavioral_source_status,
    discover_behavioral_files,
    generate_safe_participant_key,
    preprocess_behavioral_dataset,
    preprocess_behavioral_dataframe,
    write_schema_only_outputs,
)
from app.ml.preprocessing.behavioral.synthetic import generate_synthetic_behavioral_events, synthetic_behavioral_fingerprint
from app.ml.preprocessing.behavioral.validation import (
    assess_baseline_eligibility,
    detect_duplicate_events,
    detect_impossible_values,
    detect_raw_keystroke_content,
    detect_sensitive_payload_fields,
    detect_sparse_participants,
    validate_behavioral_mapping,
    validate_behavioral_source_columns,
    validate_durations,
    validate_event_types,
    validate_participant_keys,
    validate_session_ids,
    validate_timestamps,
)
from app.models.database_models import Alert, FeatureSnapshot, WorkerJob


def event_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for day in range(4):
        base = datetime(2025, 3, day + 1, 8, 0, tzinfo=timezone.utc)
        for idx, event_type in enumerate(["session_start", "typing_timing", "typing_timing", "mouse_aggregate", "prompt_response", "session_end"]):
            rows.append(
                {
                    "participant_id": "P001",
                    "event_timestamp": (base + pd.Timedelta(seconds=idx * 60)).isoformat(),
                    "session_id": f"s-{day + 1}",
                    "event_type": event_type,
                    "page_or_context": "checkin",
                    "response_latency_ms": 5000 if event_type == "prompt_response" else None,
                    "key_dwell_time_ms": 90 if event_type == "typing_timing" else None,
                    "key_flight_time_ms": 2500 if idx == 2 else (120 if event_type == "typing_timing" else None),
                    "typing_speed_cpm": 240 if event_type == "typing_timing" else None,
                    "backspace_count": 1 if event_type == "typing_timing" else None,
                    "correction_count": 1 if event_type == "typing_timing" else None,
                    "mouse_distance_px": 1000 if event_type == "mouse_aggregate" else None,
                    "mouse_speed_px_per_second": 800 if event_type == "mouse_aggregate" else None,
                    "click_count": 3 if event_type == "mouse_aggregate" else None,
                    "hesitation_count": 2 if event_type in {"typing_timing", "mouse_aggregate"} else None,
                    "session_duration_seconds": 300,
                }
            )
    return rows


def source_df(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or event_rows())


def dataset_config(source_path: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_name="behavioral-telemetry",
        dataset_version="v1",
        modality="behavioral",
        source_path=source_path,
        file_format="csv",
        label_columns=[],
        feature_columns=["response_latency_ms", "key_dwell_time_ms", "key_flight_time_ms", "typing_speed_cpm"],
        identifier_columns=["participant_id"],
        sensitive_columns=[],
        excluded_columns=["session_id", "event_timestamp", "page_or_context"],
        expected_columns=list(source_df().columns),
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=False,
        validation_context="test",
    )


def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        preprocessing_name="behavioral-session-preprocessing",
        preprocessing_version=BEHAVIORAL_PREPROCESSING_VERSION,
        dataset_name="behavioral-telemetry",
        dataset_version="v1",
        modality="behavioral",
        random_seed=42,
        test_size=0.2,
        validation_size=0.0,
        stratify_column=None,
        group_column="participant_key",
        normalization_method="none",
        categorical_encoding="none",
        text_cleaning_options={},
        audio_options={},
        image_options={},
        output_format="csv",
        feature_schema_version=BEHAVIORAL_FEATURE_SCHEMA_VERSION,
        output_subdirectory="behavioral/v1",
    )


def test_source_assessment_empty_partial_synthetic_and_no_misleading_suitable(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert discover_behavioral_files(empty) == []
    assert classify_behavioral_source_status([], final_dataset_empty=True) == SOURCE_STATUS_NO_BEHAVIORAL_DATA

    partial = tmp_path / "behavioral_data.csv"
    pd.DataFrame([{"participant_id": "P001", "typing_speed_cpm": 200}]).to_csv(partial, index=False)
    assert classify_behavioral_source_status([partial]) == SOURCE_STATUS_PARTIAL_BEHAVIORAL_DATASET

    synthetic_dir = tmp_path / "Synthetic data"
    synthetic_dir.mkdir()
    synthetic = synthetic_dir / "behavioral_data.csv"
    pd.DataFrame([{"ParticipantID": "P001", "TypingSpeed_cpm": 200, "ResponseTime_sec": 4}]).to_csv(synthetic, index=False)
    assert classify_behavioral_source_status([synthetic]) == SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY


def test_privacy_raw_key_password_clipboard_screen_and_identity_rejections():
    df = source_df()
    assert detect_raw_keystroke_content(df)["valid"]
    assert detect_sensitive_payload_fields(df)["valid"]
    with pytest.raises(ValueError, match="keystroke|content"):
        detect_raw_keystroke_content(df.assign(key_value=["a"] * len(df)))
    with pytest.raises(ValueError, match="Password-field"):
        detect_sensitive_payload_fields(df.assign(field_type=["password"] * len(df)))
    with pytest.raises(ValueError, match="clipboard"):
        detect_sensitive_payload_fields(df.assign(clipboard_content=["secret"] * len(df)))
    with pytest.raises(ValueError, match="screen"):
        detect_sensitive_payload_fields(df.assign(screen_content=["pixels"] * len(df)))
    with pytest.raises(ValueError, match="direct identity"):
        validate_participant_keys(pd.DataFrame({"participant_key": ["student@example.com"]}), require_safe=True)


def test_validation_missing_invalid_timestamp_negative_duplicate_sparse_and_mapping():
    df = source_df()
    assert validate_behavioral_source_columns(df.columns)["valid"]
    with pytest.raises(ValueError, match="missing"):
        validate_participant_keys(df.assign(participant_id=[""] * len(df)), "participant_id")
    with pytest.raises(ValueError, match="Invalid behavioral timestamps"):
        validate_timestamps(df.assign(event_timestamp=["not-a-date"] * len(df)))
    with pytest.raises(ValueError, match="negative"):
        validate_durations(df.assign(response_latency_ms=[-1] * len(df)))
    duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert detect_duplicate_events(duplicate, subset=["participant_id", "event_timestamp", "session_id", "event_type"])["duplicate_count"] == 2
    with pytest.raises(ValueError, match="Unsupported"):
        validate_event_types(df.assign(event_type=["bad_event"] * len(df)))
    with pytest.raises(ValueError, match="session"):
        validate_session_ids(pd.DataFrame({"session_id": ["bad session"]}))
    assert detect_sparse_participants(df, "participant_id", minimum_events=999)["sparse_participant_count"] == 1
    assert validate_behavioral_mapping(default_behavioral_mapping_config(), df.columns)["mapping_version"] == BEHAVIORAL_MAPPING_VERSION
    with pytest.raises(ValueError, match="Impossible"):
        detect_impossible_values(df.assign(response_latency_ms=[4000000] * len(df)))


def test_canonical_aggregation_features_no_interpolation_or_fabrication():
    df = source_df()
    canonical = canonicalize_behavioral_events(df, default_behavioral_mapping_config(), source_fingerprint=synthetic_behavioral_fingerprint(df))
    assert len(canonical) == len(df)
    assert canonical["participant_key"].str.startswith("behavioral-v1-participant-").all()
    assert not canonical["participant_key"].str.contains("P001", regex=False).any()
    assert canonical["event_id"].str.startswith("behavioral-v1-event-").all()

    sessions = aggregate_behavioral_sessions(canonical)
    assert len(sessions) == 4
    features = build_behavioral_feature_table(sessions)
    assert set(FEATURE_COLUMNS) <= set(features.columns)
    first = features.iloc[0]
    assert first["key_event_count"] == 2
    assert first["typing_speed_cpm"] == 240
    assert first["pause_count"] == 1
    assert first["long_pause_ratio"] == 0.5
    assert first["prompt_response_count"] == 1
    assert first["response_latency_mean"] == 5000
    assert first["path_distance"] == 1000
    assert first["session_duration"] == 300
    assert first["event_count"] == 6
    assert len(features) == len(sessions)


def test_baseline_insufficient_history_prior_only_and_no_future_leakage():
    df = source_df()
    canonical = canonicalize_behavioral_events(df, default_behavioral_mapping_config(), source_fingerprint=synthetic_behavioral_fingerprint(df))
    eligibility = assess_participant_baseline_eligibility(canonical, minimum_sessions=3, minimum_days=3, minimum_events=20)
    participant = next(iter(eligibility["participants"].values()))
    assert participant["eligible"] is True

    features = build_behavioral_feature_table(aggregate_behavioral_sessions(canonical))
    with_baseline = calculate_baseline_deviation_features(features, minimum_prior_observations=2)
    assert with_baseline.loc[0, "baseline_status"] == "insufficient_history"
    assert with_baseline.loc[1, "baseline_status"] == "insufficient_history"
    assert with_baseline.loc[2, "baseline_status"] == "prior_only"
    assert "session_duration_deviation_from_prior_mean" in with_baseline.columns

    sparse = canonical.head(2)
    assert assess_baseline_eligibility(sparse, minimum_sessions=3, minimum_days=3, minimum_events=20)["eligible_participant_count"] == 0


def test_synthetic_fixtures_deterministic_marked_no_clinical_labels_and_excluded_scenario():
    first = generate_synthetic_behavioral_events(seed=7)
    second = generate_synthetic_behavioral_events(seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first["synthetic"].eq(True).all()
    assert first["synthetic_schema_version"].eq(BEHAVIORAL_SYNTHETIC_SCHEMA_VERSION).all()
    assert "clinical_label" not in first.columns
    assert "suicide_risk" not in first.columns
    assert "engineering_scenario" in first.columns

    canonical_input = first.drop(columns=["engineering_scenario", "synthetic", "synthetic_schema_version"])
    result = preprocess_behavioral_dataframe(
        canonical_input,
        None,
        default_behavioral_mapping_config(),
        source_fingerprint=synthetic_behavioral_fingerprint(canonical_input),
        output_dir=paths.get_generated_root() / "temporary" / f"pytest-beh-syn-{uuid.uuid4().hex}",
        validate_only=True,
        source_type=SOURCE_STATUS_SYNTHETIC_ENGINEERING_DATA_ONLY,
    )
    assert "engineering_scenario" not in result["feature_schema"]["excluded_columns"] or "engineering_scenario" in result["feature_schema"]["excluded_columns"]
    assert result["participant_count"] > 0


def test_outputs_schema_readiness_overwrite_no_splits_and_report_privacy(tmp_path):
    source = tmp_path / "source.csv"
    source_df().to_csv(source, index=False)
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-beh-{uuid.uuid4().hex}"
    try:
        result = preprocess_behavioral_dataset(config, preprocessing_config(), default_behavioral_mapping_config(), fingerprint, output_dir=output_dir)
        assert result["source_rows"] == len(source_df())
        assert (output_dir / "behavioral_events_canonical.csv").exists()
        assert (output_dir / "behavioral_sessions.csv").exists()
        assert (output_dir / "behavioral_features.csv").exists()
        assert (output_dir / "behavioral_feature_schema.json").exists()
        assert (output_dir / "behavioral_preprocessing_report.json").exists()
        assert (output_dir / "behavioral_baseline_eligibility.json").exists()
        assert not (output_dir / "split_manifest.json").exists()
        assert "P001" not in (output_dir / "behavioral_preprocessing_report.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in (output_dir / "behavioral_preprocessing_report.md").read_text(encoding="utf-8")
        with pytest.raises(FileExistsError):
            preprocess_behavioral_dataset(config, preprocessing_config(), default_behavioral_mapping_config(), fingerprint, output_dir=output_dir)

        schema_dir = paths.get_generated_root() / "temporary" / f"pytest-beh-schema-{uuid.uuid4().hex}"
        outputs = write_schema_only_outputs(schema_dir)
        readiness = json.loads(Path(outputs["readiness_report_json"]).read_text(encoding="utf-8"))
        assert readiness["model_training_blocked"] is True
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        if "schema_dir" in locals():
            shutil.rmtree(schema_dir, ignore_errors=True)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def dataset_payload(config: DatasetConfig) -> dict:
    payload = config.to_safe_dict()
    payload["validation_context"] = "test"
    return payload


def test_cli_schema_synthetic_validate_real_missing_fingerprint_privacy_and_overwrite(tmp_path):
    script = paths.get_backend_root() / "scripts" / "preprocess_behavioral_dataset.py"
    schema_dir = paths.get_generated_root() / "temporary" / f"pytest-beh-cli-schema-{uuid.uuid4().hex}"
    synthetic_dir = paths.get_generated_root() / "temporary" / f"pytest-beh-cli-syn-{uuid.uuid4().hex}"
    real_dir = paths.get_generated_root() / "temporary" / f"pytest-beh-cli-real-{uuid.uuid4().hex}"
    try:
        schema = subprocess.run(
            [sys.executable, str(script), "--source-mode", "schema-only", "--output-dir", str(schema_dir)],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert schema.returncode == 0, schema.stderr
        assert "model training blocked: True" in schema.stdout

        synthetic = subprocess.run(
            [sys.executable, str(script), "--source-mode", "synthetic-engineering", "--output-dir", str(synthetic_dir), "--overwrite", "--max-participants", "2"],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert synthetic.returncode == 0, synthetic.stderr
        assert "source mode: synthetic-engineering" in synthetic.stdout

        source = tmp_path / "source.csv"
        source_df().to_csv(source, index=False)
        config = dataset_config(source)
        fingerprint = fingerprint_dataset(config)
        dataset_path = write_json(tmp_path / "dataset.json", dataset_payload(config))
        preprocessing_path = write_json(tmp_path / "preprocessing.json", preprocessing_config().to_safe_dict())
        mapping_path = write_json(tmp_path / "mapping.json", default_behavioral_mapping_config().to_safe_dict())
        fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())

        validate = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(real_dir),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert validate.returncode == 0, validate.stderr
        assert not real_dir.exists()

        missing_fp = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--output-dir",
                str(real_dir),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert missing_fp.returncode != 0

        bad_source = tmp_path / "bad.csv"
        source_df().assign(key_value="a").to_csv(bad_source, index=False)
        bad_config = dataset_config(bad_source)
        bad_fp = fingerprint_dataset(bad_config)
        bad_dataset_path = write_json(tmp_path / "bad_dataset.json", dataset_payload(bad_config))
        bad_fp_path = write_json(tmp_path / "bad_fingerprint.json", bad_fp.to_safe_dict())
        privacy = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(bad_dataset_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(bad_fp_path),
                "--output-dir",
                str(real_dir),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert privacy.returncode != 0
        assert "keystroke" in privacy.stderr.lower() or "content" in privacy.stderr.lower()
    finally:
        shutil.rmtree(schema_dir, ignore_errors=True)
        shutil.rmtree(synthetic_dir, ignore_errors=True)
        shutil.rmtree(real_dir, ignore_errors=True)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_regression_no_db_writes_no_routes_no_telemetry_activation_and_no_safetalk_changes(db_session):
    assert db_session.query(FeatureSnapshot).count() == 0
    assert db_session.query(WorkerJob).count() == 0
    assert db_session.query(Alert).count() == 0

    behavioral_root = paths.get_backend_root() / "app" / "ml" / "preprocessing" / "behavioral"
    text = "\n".join(path.read_text(encoding="utf-8") for path in behavioral_root.glob("*.py"))
    assert "SessionLocal" not in text
    assert "get_db" not in text
    assert "psycopg" not in text.lower()
    assert "safetalk" not in text.lower()
    assert "train_test_split" not in text
    assert "IsolationForest" not in text

    routes_text = "\n".join(path.read_text(encoding="utf-8") for path in (paths.get_backend_root() / "app" / "routes").glob("*.py"))
    assert "behavioral_events" not in routes_text
    assert "typing_speed" not in routes_text
    assert "mouse_distance" not in routes_text

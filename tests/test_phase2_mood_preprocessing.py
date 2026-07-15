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
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig, PreprocessingConfig
from app.ml.preprocessing.mood.constants import (
    FEATURE_COLUMNS,
    MOOD_FEATURE_SCHEMA_VERSION,
    MOOD_MAPPING_VERSION,
    MOOD_PREPROCESSING_VERSION,
)
from app.ml.preprocessing.mood.features import build_mood_feature_table, compute_mood_trend_score
from app.ml.preprocessing.mood.mapping import default_mood_mapping_config
from app.ml.preprocessing.mood.preprocessor import (
    canonicalize_mood_dataframe,
    create_safe_participant_key,
    generate_mood_record_id,
    normalize_timestamp,
    preprocess_mood_dataset,
    synthetic_fingerprint,
    synthetic_mood_fixture,
)
from app.ml.preprocessing.mood.validation import (
    detect_duplicate_checkins,
    detect_future_leakage,
    detect_multiple_checkins_per_period,
    detect_temporal_gaps,
    validate_mood_mapping,
    validate_mood_source_columns,
    validate_mood_value_range,
    validate_participant_keys,
    validate_timestamps,
)
from app.models.database_models import Alert, DailyCheckIn, RiskAssessment, User, UserRole
from app.routes.checkin import get_checkin_history


def fixture_rows() -> list[dict[str, object]]:
    return [
        {"ParticipantID": "P002", "Date": "2025-03-03T08:00:00Z", "Mood": 4, "CryingEpisodes": 0, "PhysicalPain": "none"},
        {"ParticipantID": "P001", "Date": "2025-03-01T08:00:00Z", "Mood": 5, "CryingEpisodes": 0, "PhysicalPain": "none"},
        {"ParticipantID": "P001", "Date": "2025-03-02T08:00:00Z", "Mood": 3, "CryingEpisodes": 1, "PhysicalPain": "none"},
        {"ParticipantID": "P001", "Date": "2025-03-04T08:00:00Z", "Mood": 1, "CryingEpisodes": 2, "PhysicalPain": "headache"},
        {"ParticipantID": "P001", "Date": "2025-03-10T08:00:00Z", "Mood": 2, "CryingEpisodes": 1, "PhysicalPain": "fatigue"},
        {"ParticipantID": "P002", "Date": "2025-03-01T18:00:00Z", "Mood": 2, "CryingEpisodes": 1, "PhysicalPain": "back"},
        {"ParticipantID": "P002", "Date": "2025-03-02T18:00:00Z", "Mood": 1, "CryingEpisodes": 1, "PhysicalPain": "none"},
    ]


def source_df(rows: list[dict[str, object]] | None = None) -> pd.DataFrame:
    return pd.DataFrame(rows or fixture_rows(), columns=["ParticipantID", "Date", "Mood", "CryingEpisodes", "PhysicalPain"])


def write_source(path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    source_df(rows).to_csv(path, index=False)
    return path


def dataset_config(source_path: Path) -> DatasetConfig:
    return DatasetConfig(
        dataset_name="daily-mood",
        dataset_version="v1",
        modality="mood",
        source_path=source_path,
        file_format="csv",
        label_columns=[],
        feature_columns=["Mood", "CryingEpisodes", "PhysicalPain"],
        identifier_columns=["ParticipantID"],
        sensitive_columns=[],
        excluded_columns=[],
        expected_columns=["ParticipantID", "Date", "Mood", "CryingEpisodes", "PhysicalPain"],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=False,
        validation_context="test",
    )


def preprocessing_config() -> PreprocessingConfig:
    return PreprocessingConfig(
        preprocessing_name="daily-mood-temporal-preprocessing",
        preprocessing_version=MOOD_PREPROCESSING_VERSION,
        dataset_name="daily-mood",
        dataset_version="v1",
        modality="mood",
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
        feature_schema_version=MOOD_FEATURE_SCHEMA_VERSION,
        output_subdirectory="mood/v1",
    )


def canonical_fixture() -> pd.DataFrame:
    return canonicalize_mood_dataframe(
        source_df(),
        default_mood_mapping_config(),
        source_fingerprint=synthetic_fingerprint(source_df()),
    )


def test_validation_missing_required_invalid_mood_timestamp_duplicate_participant_and_mapping():
    df = source_df()
    assert validate_mood_source_columns(df.columns)["valid"]
    with pytest.raises(ValueError, match="Missing required"):
        validate_mood_source_columns([column for column in df.columns if column != "Mood"])
    with pytest.raises(ValueError, match="Invalid Daily Mood mood values"):
        validate_mood_value_range(df.assign(Mood=[1, 2, 3, 4, 5, 6, 1]))
    with pytest.raises(ValueError, match="Invalid Daily Mood timestamps"):
        validate_timestamps(df.assign(Date=["not-a-date"] * len(df)))
    with pytest.raises(ValueError, match="future observations"):
        validate_timestamps(df.assign(Date=["2099-01-01"] * len(df)))
    with pytest.raises(ValueError, match="participant identifiers are missing"):
        validate_participant_keys(df.assign(ParticipantID=[""] + ["P001"] * (len(df) - 1)))

    duplicate = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert detect_duplicate_checkins(duplicate)["duplicate_count"] == 2
    assert detect_multiple_checkins_per_period(df)["allowed_by_current_production_schema"] is True
    assert validate_mood_mapping(default_mood_mapping_config())["mapping_version"] == MOOD_MAPPING_VERSION


def test_canonicalization_sorting_ids_privacy_missing_days_and_timezone():
    df = source_df()
    fp = synthetic_fingerprint(df)
    canonical = canonicalize_mood_dataframe(df, default_mood_mapping_config(), source_fingerprint=fp)

    assert canonical["participant_key"].is_monotonic_increasing
    assert not canonical["participant_key"].astype(str).str.contains("P001|P002", regex=True).any()
    assert create_safe_participant_key("P001").startswith("mood-participant-v1-")
    second = canonicalize_mood_dataframe(df, default_mood_mapping_config(), source_fingerprint=fp)
    pd.testing.assert_series_equal(canonical["record_id"], second["record_id"])
    assert canonical["record_id"].str.startswith("daily-mood-v1-").all()
    assert normalize_timestamp("2025-03-01").tzinfo is not None
    assert canonical["timestamp"].dt.tz is not None
    assert len(canonical) == len(df)
    assert 2.0 in canonical["physical_symptom_count"].fillna(-1).unique() or canonical["physical_symptom_count"].max() == 1.0
    assert canonical["timestamp"].dt.date.nunique() < 10


def test_temporal_features_values_insufficient_history_and_no_future_leakage():
    canonical = canonical_fixture()
    features = build_mood_feature_table(canonical)
    p1 = features[features["participant_key"] == create_safe_participant_key("P001")].reset_index(drop=True)

    assert set(FEATURE_COLUMNS) <= set(features.columns)
    assert pd.isna(p1.loc[0, "previous_mood"])
    assert p1.loc[1, "previous_mood"] == 5
    assert p1.loc[1, "mood_change_from_previous"] == -2
    assert p1.loc[2, "rolling_mean_3_observations"] == 3
    assert round(p1.loc[3, "rolling_mean_7_observations"], 2) == 2.75
    assert pd.isna(p1.loc[0, "rolling_std_3_observations"])
    assert p1.loc[2, "slope_last_3_observations"] < 0
    assert p1.loc[2, "consecutive_low_mood_count"] == 1
    assert p1.loc[3, "consecutive_low_mood_count"] == 2
    assert p1.loc[3, "low_mood_ratio_last_7_observations"] == 0.5
    assert p1.loc[3, "days_since_previous_checkin"] == 6
    assert p1.loc[3, "checkins_last_7_days"] == 2
    assert p1.loc[3, "missing_day_ratio_last_7_days"] > 0
    assert bool(p1.loc[1, "sudden_deterioration_flag"]) is True
    assert detect_future_leakage(features)["valid"]

    second = build_mood_feature_table(canonical)
    pd.testing.assert_frame_equal(features, second)


def test_trend_score_bounded_deterministic_higher_deterioration_and_no_risk_labels():
    low = {
        "current_mood": 5,
        "mood_change_from_previous": 1,
        "low_mood_ratio_last_7_observations": 0,
        "missing_day_ratio_last_7_days": 0,
        "sudden_deterioration_flag": False,
        "slope_last_7_observations": 0.5,
    }
    high = {
        "current_mood": 1,
        "mood_change_from_previous": -3,
        "low_mood_ratio_last_7_observations": 1,
        "missing_day_ratio_last_7_days": 0.5,
        "sudden_deterioration_flag": True,
        "slope_last_7_observations": -1,
    }
    assert 0 <= compute_mood_trend_score(low) <= 100
    assert 0 <= compute_mood_trend_score(high) <= 100
    assert compute_mood_trend_score(high) > compute_mood_trend_score(low)
    assert compute_mood_trend_score(high) == compute_mood_trend_score(high)
    features = build_mood_feature_table(canonical_fixture())
    blocked = {"risk_level", "alert", "recommendation"}
    assert not (blocked & set(features.columns))


def test_outputs_schema_overwrite_privacy_and_report(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-mood-{uuid.uuid4().hex}"
    try:
        result = preprocess_mood_dataset(
            config,
            preprocessing_config(),
            default_mood_mapping_config(),
            fingerprint,
            output_dir=output_dir,
        )
        assert result["source_rows"] == 7
        assert result["output_rows"] == 7
        assert result["participant_count"] == 2
        assert paths.is_path_inside(paths.get_generated_root(), output_dir)
        assert not paths.is_path_inside(paths.get_raw_dataset_root(), output_dir)
        assert (output_dir / "canonical_mood.csv").exists()
        assert (output_dir / "mood_features.csv").exists()
        assert (output_dir / "mood_feature_schema.json").exists()
        assert (output_dir / "mood_preprocessing_report.json").exists()
        assert (output_dir / "mood_preprocessing_report.md").exists()
        assert (output_dir / "mood_record_manifest.json").exists()
        assert "P001" not in (output_dir / "mood_preprocessing_report.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in (output_dir / "mood_preprocessing_report.md").read_text(encoding="utf-8")
        canonical = pd.read_csv(output_dir / "canonical_mood.csv")
        features = pd.read_csv(output_dir / "mood_features.csv")
        assert "mood_trend_score_0_100" in features.columns
        assert "current_mood" not in canonical.columns
        assert "mood_value" not in features.columns
        schema = json.loads((output_dir / "mood_feature_schema.json").read_text(encoding="utf-8"))
        assert schema["feature_schema_version"] == MOOD_FEATURE_SCHEMA_VERSION

        with pytest.raises(FileExistsError):
            preprocess_mood_dataset(config, preprocessing_config(), default_mood_mapping_config(), fingerprint, output_dir=output_dir)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")
    return path


def dataset_payload(source_path: Path) -> dict:
    payload = dataset_config(source_path).to_safe_dict()
    payload["validation_context"] = "test"
    return payload


def test_cli_validate_preprocess_fingerprint_missing_source_synthetic_and_overwrite(tmp_path):
    source = write_source(tmp_path / "source.csv")
    config = dataset_config(source)
    fingerprint = fingerprint_dataset(config)
    dataset_path = write_json(tmp_path / "dataset.json", dataset_payload(source))
    preprocessing_path = write_json(tmp_path / "preprocessing.json", preprocessing_config().to_safe_dict())
    mapping_path = write_json(tmp_path / "mapping.json", default_mood_mapping_config().to_safe_dict())
    fingerprint_path = write_json(tmp_path / "fingerprint.json", fingerprint.to_safe_dict())
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-mood-cli-{uuid.uuid4().hex}"
    script = paths.get_backend_root() / "scripts" / "preprocess_mood_dataset.py"
    try:
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
                str(output_dir),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert validate.returncode == 0, validate.stderr
        assert not output_dir.exists()

        preprocess = subprocess.run(
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
                str(output_dir),
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert preprocess.returncode == 0, preprocess.stderr
        assert "participants: 2" in preprocess.stdout

        overwrite = subprocess.run(
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
                str(output_dir),
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert overwrite.returncode != 0
        assert "overwrite" in overwrite.stderr.lower()

        source.write_text(source.read_text(encoding="utf-8").replace("P001", "P009", 1), encoding="utf-8")
        mismatch = subprocess.run(
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
                str(paths.get_generated_root() / "temporary" / f"pytest-mood-cli-{uuid.uuid4().hex}"),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert mismatch.returncode != 0
        assert "fingerprint mismatch" in mismatch.stderr.lower()

        missing_source_path = write_json(tmp_path / "missing_dataset.json", dataset_payload(tmp_path / "missing.csv"))
        missing = subprocess.run(
            [
                sys.executable,
                str(script),
                "--dataset-config",
                str(missing_source_path),
                "--preprocessing-config",
                str(preprocessing_path),
                "--mapping-config",
                str(mapping_path),
                "--fingerprint",
                str(fingerprint_path),
                "--output-dir",
                str(paths.get_generated_root() / "temporary" / f"pytest-mood-cli-{uuid.uuid4().hex}"),
                "--validate-only",
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert missing.returncode != 0

        synthetic_dir = paths.get_generated_root() / "temporary" / f"pytest-mood-synthetic-{uuid.uuid4().hex}"
        synthetic = subprocess.run(
            [
                sys.executable,
                str(script),
                "--source-mode",
                "synthetic-test",
                "--mapping-config",
                str(mapping_path),
                "--output-dir",
                str(synthetic_dir),
            ],
            cwd=paths.get_backend_root(),
            text=True,
            capture_output=True,
            check=False,
        )
        assert synthetic.returncode == 0, synthetic.stderr
        assert "synthetic test data: True" in synthetic.stdout
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
        if "synthetic_dir" in locals():
            shutil.rmtree(synthetic_dir, ignore_errors=True)


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(connection, record):
        cursor = connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_regression_daily_checkin_db_writes_history_and_no_alert_or_risk_changes(db_session):
    student = User(
        email="mood.student@example.com",
        username="moodstudent",
        full_name="Mood Student",
        hashed_password="hashed",
        role=UserRole.STUDENT,
    )
    db_session.add(student)
    db_session.commit()
    db_session.refresh(student)
    checkin = DailyCheckIn(
        user_id=student.id,
        mood=3,
        mood_description="Neutral",
        sleep_hours=7,
        exercise_minutes=20,
        social_interaction="Moderate",
        stress_level=5,
        anxiety_level=4,
        negative_thoughts=False,
        substance_use_today=False,
        self_harm_thoughts=False,
        notes="temporary test only",
        created_at=datetime.now(timezone.utc),
    )
    db_session.add(checkin)
    db_session.commit()

    history = get_checkin_history(current_user=student, db=db_session)

    assert history["total_checkins"] == 1
    assert history["records"][0]["mood"] == 3
    assert db_session.query(DailyCheckIn).count() == 1
    assert db_session.query(RiskAssessment).count() == 0
    assert db_session.query(Alert).count() == 0


def test_no_postgresql_or_safetalk_access_in_mood_preprocessing_sources():
    mood_root = paths.get_backend_root() / "app" / "ml" / "preprocessing" / "mood"
    text = "\n".join(path.read_text(encoding="utf-8") for path in mood_root.glob("*.py"))
    assert "SessionLocal" not in text
    assert "get_db" not in text
    assert "psycopg" not in text.lower()
    assert "safetalk" not in text.lower()


def test_temporal_gaps_reported_not_imputed():
    canonical = canonical_fixture()
    gaps = detect_temporal_gaps(canonical)
    assert gaps["max_gap_days"] >= 2
    features = build_mood_feature_table(canonical)
    assert len(features) == len(canonical)

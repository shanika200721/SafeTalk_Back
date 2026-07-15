from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic.v1 import ValidationError

from app.ml.common import paths
from app.ml.common.schemas import (
    CategoricalEncoding,
    DatasetConfig,
    DuplicatePolicy,
    FeatureDefinition,
    FeatureSchema,
    MissingValuePolicy,
    Modality,
    NormalizationMethod,
    OutputFormat,
    PreprocessingConfig,
    SplitManifest,
    SupportedFileFormat,
)
from app.ml.common.serialization import (
    load_dataset_config,
    load_feature_schema,
    load_preprocessing_config,
    load_split_manifest,
    save_schema_json,
)


def dataset_config_data(**overrides):
    data = {
        "dataset_name": "text-risk",
        "dataset_version": "v0",
        "modality": "text",
        "source_path": "Final Dataset/Text Classification dataset/mental_health_combined_test.csv",
        "file_format": "csv",
        "label_columns": ["status"],
        "feature_columns": ["text", "student_id", "gender"],
        "identifier_columns": ["student_id"],
        "sensitive_columns": ["gender"],
        "excluded_columns": ["Unique_ID"],
        "expected_columns": ["text", "status"],
        "missing_value_policy": "preserve",
        "duplicate_policy": "report_only",
        "notes": "test config",
    }
    data.update(overrides)
    return data


def preprocessing_config_data(**overrides):
    data = {
        "preprocessing_name": "text-cleaning",
        "preprocessing_version": "v0",
        "dataset_name": "text-risk",
        "dataset_version": "v0",
        "modality": "text",
        "random_seed": 42,
        "test_size": 0.2,
        "validation_size": 0.1,
        "stratify_column": "status",
        "group_column": None,
        "normalization_method": "none",
        "categorical_encoding": "none",
        "text_cleaning_options": {"lowercase": True},
        "audio_options": {},
        "image_options": {},
        "output_format": "csv",
        "feature_schema_version": "features-v0",
        "output_subdirectory": "text-risk/v0",
        "notes": "test config",
    }
    data.update(overrides)
    return data


def split_manifest_data(**overrides):
    data = {
        "dataset_name": "text-risk",
        "dataset_version": "v0",
        "preprocessing_name": "text-cleaning",
        "preprocessing_version": "v0",
        "feature_schema_version": "features-v0",
        "modality": "text",
        "random_seed": 42,
        "train_ids": ["1", "2"],
        "validation_ids": ["3"],
        "test_ids": ["4"],
        "split_created_at": datetime(2026, 7, 14, tzinfo=timezone.utc),
        "source_hash": "sourcehash",
        "config_hash": "confighash",
    }
    data.update(overrides)
    return data


def feature_schema_data(**overrides):
    data = {
        "schema_name": "text-risk-features",
        "feature_schema_version": "features-v0",
        "dataset_name": "text-risk",
        "dataset_version": "v0",
        "preprocessing_version": "v0",
        "modality": "text",
        "features": [
            {
                "name": "text_length",
                "dtype": "float",
                "description": "Length of cleaned text",
                "source_columns": ["text"],
                "nullable": False,
                "minimum": 0,
                "maximum": 100000,
                "preprocessing_step": "text_stats",
            },
            {
                "name": "sentiment_score",
                "dtype": "float",
                "description": "Derived sentiment feature",
                "source_columns": ["text"],
                "nullable": True,
                "minimum": -1,
                "maximum": 1,
                "preprocessing_step": "sentiment",
            },
        ],
        "target_columns": ["status"],
        "excluded_columns": ["student_id"],
        "created_at": datetime(2026, 7, 14, tzinfo=timezone.utc),
    }
    data.update(overrides)
    return data


def test_dataset_config_valid_config():
    config = DatasetConfig(**dataset_config_data())

    assert config.dataset_name == "text-risk"
    assert config.modality == Modality.TEXT
    assert config.expected_columns == ["text", "status", "student_id", "gender"]


def test_dataset_config_invalid_modality():
    with pytest.raises(ValidationError):
        DatasetConfig(**dataset_config_data(modality="video"))


def test_dataset_config_blank_dataset_version():
    with pytest.raises(ValidationError, match="dataset_version"):
        DatasetConfig(**dataset_config_data(dataset_version=" "))


def test_dataset_config_rejects_generated_source():
    with pytest.raises(ValidationError, match="generated"):
        DatasetConfig(
            **dataset_config_data(
                source_path="generated/preprocessing/features.csv",
                is_raw_source=False,
            )
        )


def test_dataset_config_rejects_model_source():
    with pytest.raises(ValidationError, match="ml_models"):
        DatasetConfig(
            **dataset_config_data(
                source_path="ml_models/text/model.joblib",
                is_raw_source=False,
            )
        )


def test_dataset_config_rejects_path_traversal():
    with pytest.raises(ValidationError, match="traversal"):
        DatasetConfig(**dataset_config_data(source_path="../unsafe.csv"))


def test_dataset_config_rejects_label_excluded_overlap():
    with pytest.raises(ValidationError, match="overlap"):
        DatasetConfig(**dataset_config_data(excluded_columns=["status"]))


def test_dataset_config_identifier_and_sensitive_not_ml_features():
    config = DatasetConfig(**dataset_config_data())

    assert config.ml_feature_columns() == ["text"]


def test_dataset_config_serialization_round_trip():
    config = DatasetConfig(**dataset_config_data())
    payload = config.to_safe_dict()
    restored = DatasetConfig.parse_obj(payload)

    assert restored == config


def test_dataset_config_source_existence_validation(tmp_path):
    temp_source = tmp_path / "source.csv"
    temp_source.write_text("x,y\n1,2\n", encoding="utf-8")
    config = DatasetConfig(
        **dataset_config_data(
            source_path=temp_source,
            validation_context="test",
        )
    )

    assert config.validate_source_exists() == temp_source.resolve()

    missing = DatasetConfig(
        **dataset_config_data(
            source_path=tmp_path / "missing.csv",
            validation_context="test",
        )
    )
    with pytest.raises(FileNotFoundError):
        missing.validate_source_exists()


def test_preprocessing_config_valid_config():
    config = PreprocessingConfig(**preprocessing_config_data())

    assert config.preprocessing_name == "text-cleaning"
    assert config.output_format == OutputFormat.CSV


def test_preprocessing_config_deterministic_train_size():
    config = PreprocessingConfig(**preprocessing_config_data(test_size=0.25, validation_size=0.15))

    assert config.train_size == 0.6
    assert config.split_percentages() == {"train": 0.6, "validation": 0.15, "test": 0.25}


def test_preprocessing_config_invalid_test_size():
    with pytest.raises(ValidationError, match="test_size"):
        PreprocessingConfig(**preprocessing_config_data(test_size=0))


def test_preprocessing_config_invalid_validation_size():
    with pytest.raises(ValidationError, match="validation_size"):
        PreprocessingConfig(**preprocessing_config_data(validation_size=-0.1))


def test_preprocessing_config_invalid_combined_split_size():
    with pytest.raises(ValidationError, match="less than 1"):
        PreprocessingConfig(**preprocessing_config_data(test_size=0.6, validation_size=0.4))


def test_preprocessing_config_negative_random_seed():
    with pytest.raises(ValidationError, match="random_seed"):
        PreprocessingConfig(**preprocessing_config_data(random_seed=-1))


def test_preprocessing_config_absolute_output_path_rejection(tmp_path):
    with pytest.raises(ValidationError, match="relative"):
        PreprocessingConfig(**preprocessing_config_data(output_subdirectory=tmp_path))


def test_preprocessing_config_traversal_rejection():
    with pytest.raises(ValidationError, match="traversal"):
        PreprocessingConfig(**preprocessing_config_data(output_subdirectory="../unsafe"))


def test_preprocessing_config_output_resolves_under_generated_preprocessing():
    config = PreprocessingConfig(**preprocessing_config_data())

    assert paths.is_path_inside(paths.get_generated_preprocessing_root(), config.resolved_output_path())


def test_preprocessing_config_modality_specific_option_validation():
    with pytest.raises(ValidationError, match="text_cleaning_options"):
        PreprocessingConfig(
            **preprocessing_config_data(
                modality="profile",
                text_cleaning_options={"lowercase": True},
            )
        )

    voice_config = PreprocessingConfig(
        **preprocessing_config_data(
            modality="voice",
            text_cleaning_options={},
            audio_options={"sample_rate": 16000},
        )
    )
    assert voice_config.audio_options == {"sample_rate": 16000}

    with pytest.raises(ValidationError, match="audio_options"):
        PreprocessingConfig(
            **preprocessing_config_data(
                modality="text",
                audio_options={"sample_rate": 16000},
            )
        )

    with pytest.raises(ValidationError, match="image_options"):
        PreprocessingConfig(
            **preprocessing_config_data(
                modality="text",
                image_options={"size": 48},
            )
        )


def test_split_manifest_valid_manifest():
    manifest = SplitManifest(**split_manifest_data())

    assert manifest.total_records() == 4
    assert manifest.split_counts() == {"train": 2, "validation": 1, "test": 1}
    assert manifest.contains_id(1)


def test_split_manifest_duplicate_id_within_split():
    with pytest.raises(ValidationError, match="duplicate"):
        SplitManifest(**split_manifest_data(train_ids=["1", "1"]))


def test_split_manifest_overlap_between_splits():
    with pytest.raises(ValidationError, match="overlap"):
        SplitManifest(**split_manifest_data(train_ids=["1"], test_ids=["1"]))


def test_split_manifest_empty_training_split():
    with pytest.raises(ValidationError, match="train_ids"):
        SplitManifest(**split_manifest_data(train_ids=[]))


def test_split_manifest_empty_test_split():
    with pytest.raises(ValidationError, match="test_ids"):
        SplitManifest(**split_manifest_data(test_ids=[]))


def test_split_manifest_timezone_naive_datetime_rejection():
    with pytest.raises(ValidationError, match="timezone-aware"):
        SplitManifest(**split_manifest_data(split_created_at=datetime(2026, 7, 14)))


def test_split_manifest_serialization_round_trip():
    manifest = SplitManifest(**split_manifest_data(train_ids=[1, 2], validation_ids=[], test_ids=[3]))
    restored = SplitManifest.parse_obj(manifest.to_safe_dict())

    assert restored.train_ids == ["1", "2"]
    assert restored.validation_ids == []
    assert restored.test_ids == ["3"]
    assert restored.split_created_at.tzinfo is not None


def test_feature_schema_valid_schema():
    schema = FeatureSchema(**feature_schema_data())

    assert schema.feature_names() == ["text_length", "sentiment_score"]


def test_feature_schema_duplicate_feature_names():
    data = feature_schema_data()
    data["features"].append(data["features"][0].copy())
    with pytest.raises(ValidationError, match="unique"):
        FeatureSchema(**data)


def test_feature_schema_target_leakage_into_features():
    data = feature_schema_data()
    data["features"][0]["name"] = "status"
    with pytest.raises(ValidationError, match="target"):
        FeatureSchema(**data)


def test_feature_schema_excluded_feature_leakage():
    data = feature_schema_data()
    data["features"][0]["name"] = "student_id"
    with pytest.raises(ValidationError, match="excluded"):
        FeatureSchema(**data)


def test_feature_schema_invalid_numeric_range():
    data = feature_schema_data()
    data["features"][0]["minimum"] = 10
    data["features"][0]["maximum"] = 1
    with pytest.raises(ValidationError, match="minimum"):
        FeatureSchema(**data)


def test_feature_schema_source_column_collection():
    schema = FeatureSchema(**feature_schema_data())

    assert schema.required_source_columns() == ["text"]


def test_feature_schema_dataframe_column_validation_without_pandas():
    schema = FeatureSchema(**feature_schema_data())

    schema.validate_dataframe_columns(["text", "status"])
    with pytest.raises(ValueError, match="Missing"):
        schema.validate_dataframe_columns(["status"])


def test_serialization_overwrite_protection(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_GENERATED_ROOT", tmp_path / "generated")
    config = DatasetConfig(**dataset_config_data())
    output = paths.get_generated_root() / "manifests" / "dataset.json"

    save_schema_json(config, output)
    with pytest.raises(FileExistsError):
        save_schema_json(config, output)

    save_schema_json(config, output, overwrite=True)
    assert output.exists()


def test_serialization_malformed_json(tmp_path):
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{bad", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        load_dataset_config(bad_json)


def test_serialization_raw_dataset_output_path_rejection():
    config = DatasetConfig(**dataset_config_data())

    with pytest.raises(ValueError, match="raw dataset"):
        save_schema_json(config, paths.get_raw_dataset_root() / "unsafe.json")


def test_serialization_deterministic_json_output(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_GENERATED_ROOT", tmp_path / "generated")
    config = DatasetConfig(**dataset_config_data())
    output_a = paths.get_generated_root() / "manifests" / "a.json"
    output_b = paths.get_generated_root() / "manifests" / "b.json"

    save_schema_json(config, output_a)
    save_schema_json(config, output_b)

    assert output_a.read_text(encoding="utf-8") == output_b.read_text(encoding="utf-8")


def test_serialization_loaders_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "_GENERATED_ROOT", tmp_path / "generated")

    dataset = DatasetConfig(**dataset_config_data())
    preprocessing = PreprocessingConfig(**preprocessing_config_data())
    manifest = SplitManifest(**split_manifest_data())
    feature_schema = FeatureSchema(**feature_schema_data())

    dataset_path = paths.get_generated_root() / "manifests" / "dataset.json"
    preprocessing_path = paths.get_generated_root() / "manifests" / "preprocessing.json"
    manifest_path = paths.get_generated_root() / "manifests" / "split.json"
    feature_schema_path = paths.get_generated_root() / "manifests" / "features.json"

    save_schema_json(dataset, dataset_path)
    save_schema_json(preprocessing, preprocessing_path)
    save_schema_json(manifest, manifest_path)
    save_schema_json(feature_schema, feature_schema_path)

    assert load_dataset_config(dataset_path).dataset_name == dataset.dataset_name
    assert load_preprocessing_config(preprocessing_path).preprocessing_name == preprocessing.preprocessing_name
    assert load_split_manifest(manifest_path).split_counts() == manifest.split_counts()
    assert load_feature_schema(feature_schema_path).feature_names() == feature_schema.feature_names()


def test_enum_values_are_exact():
    assert [item.value for item in MissingValuePolicy] == ["error", "drop_rows", "drop_columns", "impute", "preserve"]
    assert [item.value for item in DuplicatePolicy] == ["error", "keep_first", "keep_last", "remove_exact", "report_only"]
    assert [item.value for item in SupportedFileFormat][:3] == ["csv", "tsv", "json"]
    assert NormalizationMethod.STANDARD.value == "standard"
    assert CategoricalEncoding.ONE_HOT.value == "one_hot"

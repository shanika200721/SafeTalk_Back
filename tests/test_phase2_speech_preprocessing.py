from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import uuid
import wave
from pathlib import Path

import numpy as np
import pytest

from app.ml.common import paths
from app.ml.common.fingerprinting import fingerprint_dataset
from app.ml.common.schemas import DatasetConfig
from app.ml.preprocessing.speech.audio_io import extract_audio_metadata
from app.ml.preprocessing.speech.constants import FEATURE_COLUMNS
from app.ml.preprocessing.speech.features import build_speech_feature_schema, extract_acoustic_features
from app.ml.preprocessing.speech.filename_parsers import (
    parse_crema_filename,
    parse_ravdess_filename,
    parse_savee_filename,
    parse_tess_filename,
)
from app.ml.preprocessing.speech.mapping import default_speech_label_mapping_config, normalize_emotion_label
from app.ml.preprocessing.speech.preprocessor import (
    generate_safe_speaker_key,
    generate_speech_record_id,
    preprocess_speech_dataset,
)
from app.ml.preprocessing.speech.schemas import SpeechCorpusConfig, SpeechCorpusMappingConfig
from app.ml.preprocessing.speech.validation import detect_cross_corpus_duplicates, detect_audio_duplicates


def write_wav(path: Path, *, sample_rate=16000, channels=1, duration=0.4, freq=440.0, silence=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t) if silence else 0.35 * np.sin(2 * np.pi * freq * t)
    pcm = (signal * 32767).astype(np.int16)
    if channels == 2:
        pcm = np.column_stack([pcm, pcm])
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def dataset_config(source_path: Path, name: str) -> DatasetConfig:
    return DatasetConfig(
        dataset_name=name,
        dataset_version="v1",
        modality="voice",
        source_path=source_path,
        file_format="folder",
        label_columns=[],
        feature_columns=[],
        identifier_columns=[],
        sensitive_columns=[],
        excluded_columns=[],
        expected_columns=[],
        missing_value_policy="preserve",
        duplicate_policy="report_only",
        is_raw_source=True,
        validation_context="test",
    )


def corpus_mapping(root: Path) -> SpeechCorpusMappingConfig:
    return SpeechCorpusMappingConfig(
        corpora=[
            SpeechCorpusConfig(
                corpus_name="CREMA",
                source_path=str(root / "Crema"),
                filename_parser="parse_crema_filename",
                speaker_id_rule="speaker token",
                emotion_label_rule="emotion token",
                expected_file_format="wav",
                license_note="fixture",
            ),
            SpeechCorpusConfig(
                corpus_name="RAVDESS",
                source_path=str(root / "Ravdess"),
                filename_parser="parse_ravdess_filename",
                speaker_id_rule="actor token",
                emotion_label_rule="emotion token",
                expected_file_format="wav",
                license_note="fixture",
            ),
            SpeechCorpusConfig(
                corpus_name="SAVEE",
                source_path=str(root / "Savee"),
                filename_parser="parse_savee_filename",
                speaker_id_rule="speaker prefix",
                emotion_label_rule="emotion code",
                expected_file_format="wav",
                license_note="fixture",
            ),
            SpeechCorpusConfig(
                corpus_name="TESS",
                source_path=str(root / "Tess"),
                filename_parser="parse_tess_filename",
                speaker_id_rule="speaker prefix",
                emotion_label_rule="emotion suffix",
                expected_file_format="wav",
                license_note="fixture",
            ),
        ]
    )


def write_fixture_corpora(root: Path) -> None:
    write_wav(root / "Crema" / "1001_DFA_ANG_XX.wav")
    write_wav(root / "Ravdess" / "03-01-01-01-01-01-01.wav", channels=2)
    write_wav(root / "Savee" / "DC_a01.wav")
    write_wav(root / "Tess" / "OAF_back_angry.wav")


def test_filename_parsers_valid_and_invalid():
    assert parse_ravdess_filename("03-01-01-01-01-01-01.wav").speaker_id == "01"
    with pytest.raises(ValueError, match="Malformed RAVDESS"):
        parse_ravdess_filename("bad.wav")
    assert parse_crema_filename("1001_DFA_ANG_XX.wav").original_emotion_label == "ANG"
    with pytest.raises(ValueError, match="Malformed CREMA"):
        parse_crema_filename("1001_DFA_BAD_XX.wav")
    assert parse_savee_filename("DC_sa01.wav").original_emotion_label == "sa"
    with pytest.raises(ValueError, match="Malformed SAVEE"):
        parse_savee_filename("DC_x01.wav")
    assert parse_tess_filename("OAF_back_ps.wav").original_emotion_label == "ps"
    with pytest.raises(ValueError, match="Malformed TESS"):
        parse_tess_filename("OAF_back_boredom.wav")


def test_audio_metadata_valid_stereo_zero_corrupt_unsupported_and_deterministic(tmp_path):
    mono = write_wav(tmp_path / "mono.wav", sample_rate=8000, channels=1, duration=0.5)
    stereo = write_wav(tmp_path / "stereo.wav", sample_rate=16000, channels=2, duration=0.25)
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    corrupt = tmp_path / "bad.wav"
    corrupt.write_bytes(b"not a wav")
    unsupported = tmp_path / "x.ogg"
    unsupported.write_bytes(b"data")
    first = extract_audio_metadata(mono)
    second = extract_audio_metadata(mono)
    assert first == second
    assert first.sample_rate == 8000
    assert first.channel_count == 1
    assert first.duration_seconds == pytest.approx(0.5)
    assert extract_audio_metadata(stereo).channel_count == 2
    assert not extract_audio_metadata(empty).readable
    assert not extract_audio_metadata(corrupt).readable
    assert "unsupported" in "|".join(extract_audio_metadata(unsupported).validation_warnings)


def test_labels_mapping_no_silent_merging_and_target_excluded_from_features():
    mapping = default_speech_label_mapping_config()
    assert normalize_emotion_label("CREMA", "ANG", mapping) == "angry"
    assert normalize_emotion_label("SAVEE", "sa", mapping) == "sad"
    with pytest.raises(ValueError, match="Unknown"):
        normalize_emotion_label("CREMA", "BORED", mapping)
    schema = build_speech_feature_schema()
    assert "canonical_emotion_label" in schema.target_columns
    assert "canonical_emotion_label" not in schema.feature_names()


def test_deterministic_ids_change_with_fingerprint_and_no_raw_speaker_leakage():
    rid1 = generate_speech_record_id("CREMA", "Final Dataset/Speech Emotion/Crema/1001_DFA_ANG_XX.wav", "a" * 64)
    rid2 = generate_speech_record_id("CREMA", "Final Dataset/Speech Emotion/Crema/1001_DFA_ANG_XX.wav", "a" * 64)
    rid3 = generate_speech_record_id("CREMA", "Final Dataset/Speech Emotion/Crema/1001_DFA_ANG_XX.wav", "b" * 64)
    assert rid1 == rid2
    assert rid1 != rid3
    key = generate_safe_speaker_key("CREMA", "1001", "a" * 64)
    assert "1001" not in key
    assert key != generate_safe_speaker_key("CREMA", "1001", "b" * 64)


def test_features_mfcc_rms_zcr_spectral_no_nan_deterministic_short_and_corrupt(tmp_path):
    wav = write_wav(tmp_path / "a.wav", duration=0.2)
    first, warnings = extract_acoustic_features(wav)
    second, _ = extract_acoustic_features(wav)
    assert set(FEATURE_COLUMNS) == set(first)
    assert first == second
    assert first["rms_energy_mean"] > 0
    assert first["zero_crossing_rate_mean"] >= 0
    assert first["spectral_centroid_mean"] > 0
    assert "mfcc_01_mean" in first
    assert all(math.isfinite(value) for value in first.values())
    corrupt = tmp_path / "bad.wav"
    corrupt.write_bytes(b"bad")
    bad_features, bad_warnings = extract_acoustic_features(corrupt)
    assert all(math.isfinite(value) for value in bad_features.values())
    assert bad_warnings


def test_duplicates_cross_corpus_and_safe_duplicate_report(tmp_path):
    root = tmp_path / "speech"
    a = write_wav(root / "A" / "one.wav")
    b = root / "A" / "two.wav"
    b.write_bytes(a.read_bytes())
    c = root / "B" / "three.wav"
    c.parent.mkdir(parents=True, exist_ok=True)
    c.write_bytes(a.read_bytes())
    assert len(detect_audio_duplicates([a, b])) == 1
    cross = detect_cross_corpus_duplicates({"A": [a, b], "B": [c]})
    assert len(cross) == 1
    assert "one.wav" in json.dumps(cross)


def test_preprocessing_outputs_paths_overwrite_privacy_no_splits_or_model_artifacts(tmp_path):
    root = tmp_path / "speech"
    write_fixture_corpora(root)
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-speech-{uuid.uuid4().hex}"
    try:
        result = preprocess_speech_dataset(
            corpus_mapping(root),
            default_speech_label_mapping_config(),
            source_fingerprints={name: "a" * 64 for name in ["CREMA", "RAVDESS", "SAVEE", "TESS"]},
            output_dir=output_dir,
            overwrite=False,
            extract_features=True,
        )
        assert result["output_records"] == 4
        assert (output_dir / "speech_canonical_manifest.csv").exists()
        assert (output_dir / "speech_features.csv").exists()
        assert (output_dir / "speech_feature_schema.json").exists()
        assert not any((output_dir / name).exists() for name in ["train.csv", "validation.csv", "test.csv", "model.pkl", "scaler.pkl"])
        report_text = (output_dir / "speech_preprocessing_report.json").read_text(encoding="utf-8")
        assert str(tmp_path) not in report_text
        assert "1001" not in report_text
        with pytest.raises(FileExistsError):
            preprocess_speech_dataset(
                corpus_mapping(root),
                default_speech_label_mapping_config(),
                source_fingerprints={name: "a" * 64 for name in ["CREMA", "RAVDESS", "SAVEE", "TESS"]},
                output_dir=output_dir,
                overwrite=False,
            )
    finally:
        for child in output_dir.glob("*"):
            child.unlink()
        output_dir.rmdir()


def test_cli_validate_metadata_features_missing_mapping_fingerprint_and_normalized_audio(tmp_path):
    root = tmp_path / "speech"
    write_fixture_corpora(root)
    mapping = corpus_mapping(root)
    corpus_mapping_path = tmp_path / "corpus.json"
    corpus_mapping_path.write_text(json.dumps(mapping.to_safe_dict()), encoding="utf-8")
    label_path = tmp_path / "labels.json"
    label_path.write_text(json.dumps(default_speech_label_mapping_config().to_safe_dict()), encoding="utf-8")
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({"dataset_name": "fixture", "source_root": str(root)}), encoding="utf-8")
    prep_path = tmp_path / "prep.json"
    prep_path.write_text(json.dumps({"preprocessing_version": "1.0.0"}), encoding="utf-8")
    fingerprint_dir = tmp_path / "fingerprints"
    fingerprint_dir.mkdir()
    for corpus in mapping.corpora:
        config = dataset_config(Path(corpus.source_path), f"speech-emotion-{corpus.corpus_name.lower()}")
        (fingerprint_dir / f"{corpus.corpus_name.lower()}.json").write_text(
            json.dumps(fingerprint_dataset(config).to_safe_dict()),
            encoding="utf-8",
        )
    script = paths.get_backend_root() / "scripts" / "preprocess_speech_dataset.py"
    output_dir = paths.get_generated_root() / "temporary" / f"pytest-speech-cli-{uuid.uuid4().hex}"
    base = [
        sys.executable,
        str(script),
        "--dataset-config",
        str(dataset_path),
        "--preprocessing-config",
        str(prep_path),
        "--label-mapping-config",
        str(label_path),
        "--corpus-mapping-config",
        str(corpus_mapping_path),
        "--fingerprint-dir",
        str(fingerprint_dir),
        "--output-dir",
        str(output_dir),
        "--max-files",
        "1",
    ]
    validate = subprocess.run(base + ["--validate-only"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert validate.returncode == 0, validate.stderr
    metadata = subprocess.run(base + ["--overwrite"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert metadata.returncode == 0, metadata.stderr
    features = subprocess.run(base + ["--overwrite", "--extract-features"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert features.returncode == 0, features.stderr
    assert not (output_dir / "speech_normalized_audio_manifest.json").exists()
    normalized = subprocess.run(
        base + ["--overwrite", "--write-normalized-audio", "--target-sample-rate", "8000", "--mono"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert normalized.returncode == 0, normalized.stderr
    assert (output_dir / "speech_normalized_audio_manifest.json").exists()
    shutil.rmtree(paths.get_generated_preprocessing_root() / "speech" / "v1" / "audio", ignore_errors=True)
    missing_base = list(base)
    missing_base[missing_base.index("--label-mapping-config") + 1] = str(tmp_path / "missing.json")
    missing_mapping = subprocess.run(missing_base, cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert missing_mapping.returncode != 0
    changed = root / "Crema" / "1001_DFA_ANG_XX.wav"
    changed.write_bytes(changed.read_bytes() + b"x")
    mismatch = subprocess.run(base + ["--validate-only"], cwd=paths.get_backend_root(), text=True, capture_output=True, check=False)
    assert mismatch.returncode != 0
    assert "fingerprint mismatch" in mismatch.stderr.lower()

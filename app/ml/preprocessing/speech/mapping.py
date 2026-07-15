"""Versioned corpus and label mapping helpers for speech emotion datasets."""

from __future__ import annotations

import json
from pathlib import Path

from app.ml.preprocessing.speech.constants import (
    CANONICAL_EMOTION_LABELS,
    CORPUS_CREMA,
    CORPUS_RAVDESS,
    CORPUS_SAVEE,
    CORPUS_TESS,
    SPEECH_LABEL_MAPPING_VERSION,
)
from app.ml.preprocessing.speech.schemas import (
    SpeechCorpusConfig,
    SpeechCorpusMappingConfig,
    SpeechLabelMappingConfig,
    SpeechLabelMappingEntry,
)


def default_speech_corpus_mapping_config() -> SpeechCorpusMappingConfig:
    return SpeechCorpusMappingConfig(
        corpora=[
            SpeechCorpusConfig(
                corpus_name=CORPUS_CREMA,
                source_path="Final Dataset/Speech Emotion/Crema",
                filename_parser="parse_crema_filename",
                speaker_id_rule="first underscore-delimited token, e.g. 1001",
                emotion_label_rule="third underscore-delimited token: ANG, DIS, FEA, HAP, NEU, SAD",
                gender_rule=None,
                expected_file_format="wav",
                expected_sample_rate=None,
                license_note="Distributed via Kaggle speech-emotion collection; verify upstream CREMA-D license before reuse.",
                included=True,
                notes="Actor identifiers are available in filenames; no gender is encoded in local filenames.",
            ),
            SpeechCorpusConfig(
                corpus_name=CORPUS_RAVDESS,
                source_path="Final Dataset/Speech Emotion/Ravdess",
                filename_parser="parse_ravdess_filename",
                speaker_id_rule="seventh dash-delimited token, actor 01-24",
                emotion_label_rule="third dash-delimited token, code 01-08",
                gender_rule="RAVDESS actor odd/even convention: odd male, even female",
                expected_file_format="wav",
                expected_sample_rate=48000,
                license_note="RAVDESS is publicly available for research; verify original license/citation requirements.",
                included=True,
                notes="Filename also encodes modality, vocal channel, intensity, statement, and repetition.",
            ),
            SpeechCorpusConfig(
                corpus_name=CORPUS_SAVEE,
                source_path="Final Dataset/Speech Emotion/Savee",
                filename_parser="parse_savee_filename",
                speaker_id_rule="uppercase speaker prefix before underscore, e.g. DC",
                emotion_label_rule="lowercase emotion code after underscore: a, d, f, h, n, sa, su",
                gender_rule="SAVEE speakers are documented as male in the common corpus description.",
                expected_file_format="wav",
                expected_sample_rate=44100,
                license_note="SAVEE is a research corpus; verify source usage terms before redistribution.",
                included=True,
                notes="Emotion code length varies, so parsing must prefer sa/su before single-letter codes.",
            ),
            SpeechCorpusConfig(
                corpus_name=CORPUS_TESS,
                source_path="Final Dataset/Speech Emotion/Tess",
                filename_parser="parse_tess_filename",
                speaker_id_rule="prefix before first underscore: OAF or YAF",
                emotion_label_rule="final underscore-delimited token: angry, disgust, fear, happy, neutral, ps, sad",
                gender_rule="TESS OAF/YAF speakers are documented as female speakers from different age groups.",
                expected_file_format="wav",
                expected_sample_rate=24414,
                license_note="TESS is commonly shared for research; verify original terms and citation requirements.",
                included=True,
                notes="The ps label is pleasant surprise and is mapped to surprised with an explicit merge note.",
            ),
        ],
        notes="Each corpus is treated as a distinct source dataset. Corpus identity is preserved as metadata.",
    )


def _entry(corpus: str, original: str, canonical: str, notes: str, *, merged: bool = False, retained: bool = True, excluded: bool = False):
    return SpeechLabelMappingEntry(
        corpus_name=corpus,
        original_label=original,
        canonical_label=canonical,
        retained=retained,
        merged=merged,
        excluded=excluded,
        notes=notes,
    )


def default_speech_label_mapping_config() -> SpeechLabelMappingConfig:
    entries = [
        _entry(CORPUS_CREMA, "ANG", "angry", "CREMA-D anger code retained as angry."),
        _entry(CORPUS_CREMA, "DIS", "disgust", "CREMA-D disgust code retained as disgust."),
        _entry(CORPUS_CREMA, "FEA", "fearful", "CREMA-D fear code mapped to canonical fearful.", merged=True),
        _entry(CORPUS_CREMA, "HAP", "happy", "CREMA-D happiness code retained as happy."),
        _entry(CORPUS_CREMA, "NEU", "neutral", "CREMA-D neutral code retained as neutral."),
        _entry(CORPUS_CREMA, "SAD", "sad", "CREMA-D sadness code retained as sad."),
        _entry(CORPUS_RAVDESS, "01", "neutral", "RAVDESS neutral code retained as neutral."),
        _entry(CORPUS_RAVDESS, "02", "calm", "RAVDESS calm code retained as calm."),
        _entry(CORPUS_RAVDESS, "03", "happy", "RAVDESS happy code retained as happy."),
        _entry(CORPUS_RAVDESS, "04", "sad", "RAVDESS sad code retained as sad."),
        _entry(CORPUS_RAVDESS, "05", "angry", "RAVDESS angry code retained as angry."),
        _entry(CORPUS_RAVDESS, "06", "fearful", "RAVDESS fearful code retained as fearful."),
        _entry(CORPUS_RAVDESS, "07", "disgust", "RAVDESS disgust code retained as disgust."),
        _entry(CORPUS_RAVDESS, "08", "surprised", "RAVDESS surprised code retained as surprised."),
        _entry(CORPUS_SAVEE, "a", "angry", "SAVEE anger code retained as angry."),
        _entry(CORPUS_SAVEE, "d", "disgust", "SAVEE disgust code retained as disgust."),
        _entry(CORPUS_SAVEE, "f", "fearful", "SAVEE fear code mapped to canonical fearful.", merged=True),
        _entry(CORPUS_SAVEE, "h", "happy", "SAVEE happiness code retained as happy."),
        _entry(CORPUS_SAVEE, "n", "neutral", "SAVEE neutral code retained as neutral."),
        _entry(CORPUS_SAVEE, "sa", "sad", "SAVEE sadness code retained as sad."),
        _entry(CORPUS_SAVEE, "su", "surprised", "SAVEE surprise code mapped to canonical surprised.", merged=True),
        _entry(CORPUS_TESS, "angry", "angry", "TESS angry label retained as angry."),
        _entry(CORPUS_TESS, "disgust", "disgust", "TESS disgust label retained as disgust."),
        _entry(CORPUS_TESS, "fear", "fearful", "TESS fear label mapped to canonical fearful.", merged=True),
        _entry(CORPUS_TESS, "happy", "happy", "TESS happy label retained as happy."),
        _entry(CORPUS_TESS, "neutral", "neutral", "TESS neutral label retained as neutral."),
        _entry(CORPUS_TESS, "ps", "surprised", "TESS ps means pleasant surprise; retained as surprised with this caveat.", merged=True),
        _entry(CORPUS_TESS, "sad", "sad", "TESS sad label retained as sad."),
    ]
    return SpeechLabelMappingConfig(
        mapping_version=SPEECH_LABEL_MAPPING_VERSION,
        canonical_labels=list(CANONICAL_EMOTION_LABELS),
        entries=entries,
        notes="Mappings are versioned and corpus-specific; labels are not silently merged across corpora.",
    )


def load_speech_label_mapping_config(path: str | Path | None) -> SpeechLabelMappingConfig:
    if path is None:
        return default_speech_label_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Speech label mapping config must be a JSON object")
    return SpeechLabelMappingConfig.parse_obj(payload)


def load_speech_corpus_mapping_config(path: str | Path | None) -> SpeechCorpusMappingConfig:
    if path is None:
        return default_speech_corpus_mapping_config()
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Speech corpus mapping config must be a JSON object")
    return SpeechCorpusMappingConfig.parse_obj(payload)


def normalize_emotion_label(corpus_name: str, original_label: object, config: SpeechLabelMappingConfig) -> str:
    value = "" if original_label is None else str(original_label).strip()
    for entry in config.entries:
        if entry.corpus_name == corpus_name and entry.original_label == value:
            if not entry.retained or entry.excluded:
                raise ValueError(f"Speech label is excluded by mapping: {corpus_name}:{value}")
            return entry.canonical_label
    raise ValueError(f"Unknown speech emotion label: {corpus_name}:{value}")


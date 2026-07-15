"""Corpus-specific filename parsers for speech emotion datasets."""

from __future__ import annotations

import re
from pathlib import Path

from app.ml.preprocessing.speech.constants import CORPUS_CREMA, CORPUS_RAVDESS, CORPUS_SAVEE, CORPUS_TESS
from app.ml.preprocessing.speech.schemas import SpeechSourceRecord

_CREMA_RE = re.compile(r"^(?P<speaker>\d{4})_(?P<statement>[A-Z]{3})_(?P<emotion>ANG|DIS|FEA|HAP|NEU|SAD)_(?P<intensity>XX|LO|MD|HI)\.wav$", re.IGNORECASE)
_RAVDESS_RE = re.compile(
    r"^(?P<modality>0[123])-(?P<vocal_channel>0[12])-(?P<emotion>0[1-8])-(?P<intensity>0[12])-(?P<statement>0[12])-(?P<repetition>0[12])-(?P<actor>\d{2})\.wav$",
    re.IGNORECASE,
)
_SAVEE_RE = re.compile(r"^(?P<speaker>DC|JE|JK|KL)_(?P<emotion>sa|su|a|d|f|h|n)(?P<repetition>\d{2})\.wav$", re.IGNORECASE)
_TESS_RE = re.compile(r"^(?P<speaker>OAF|YAF)_(?P<word>[A-Za-z]+)_(?P<emotion>angry|disgust|fear|happy|neutral|ps|sad)\.wav$", re.IGNORECASE)


def _relative_source_file(path: str | Path) -> str:
    return Path(path).name.replace("\\", "/")


def parse_crema_filename(path: str | Path) -> SpeechSourceRecord:
    name = Path(path).name
    match = _CREMA_RE.match(name)
    if not match:
        raise ValueError(f"Malformed CREMA filename: {name}")
    return SpeechSourceRecord(
        source_file=_relative_source_file(path),
        corpus_name=CORPUS_CREMA,
        speaker_id=match.group("speaker"),
        original_emotion_label=match.group("emotion").upper(),
        intensity=match.group("intensity").upper(),
        statement_id=match.group("statement").upper(),
    )


def parse_ravdess_filename(path: str | Path) -> SpeechSourceRecord:
    name = Path(path).name
    match = _RAVDESS_RE.match(name)
    if not match:
        raise ValueError(f"Malformed RAVDESS filename: {name}")
    actor = match.group("actor")
    actor_number = int(actor)
    if actor_number < 1 or actor_number > 24:
        raise ValueError(f"Malformed RAVDESS filename actor code: {name}")
    return SpeechSourceRecord(
        source_file=_relative_source_file(path),
        corpus_name=CORPUS_RAVDESS,
        speaker_id=actor,
        original_emotion_label=match.group("emotion"),
        gender="male" if actor_number % 2 == 1 else "female",
        intensity=match.group("intensity"),
        statement_id=match.group("statement"),
        repetition_id=match.group("repetition"),
    )


def parse_savee_filename(path: str | Path) -> SpeechSourceRecord:
    name = Path(path).name
    match = _SAVEE_RE.match(name)
    if not match:
        raise ValueError(f"Malformed SAVEE filename: {name}")
    return SpeechSourceRecord(
        source_file=_relative_source_file(path),
        corpus_name=CORPUS_SAVEE,
        speaker_id=match.group("speaker").upper(),
        original_emotion_label=match.group("emotion").lower(),
        gender="male",
        repetition_id=match.group("repetition"),
    )


def parse_tess_filename(path: str | Path) -> SpeechSourceRecord:
    name = Path(path).name
    match = _TESS_RE.match(name)
    if not match:
        raise ValueError(f"Malformed TESS filename: {name}")
    return SpeechSourceRecord(
        source_file=_relative_source_file(path),
        corpus_name=CORPUS_TESS,
        speaker_id=match.group("speaker").upper(),
        original_emotion_label=match.group("emotion").lower(),
        gender="female",
        statement_id=match.group("word").lower(),
    )


PARSER_BY_NAME = {
    "parse_crema_filename": parse_crema_filename,
    "parse_ravdess_filename": parse_ravdess_filename,
    "parse_savee_filename": parse_savee_filename,
    "parse_tess_filename": parse_tess_filename,
}


def parse_with_named_parser(parser_name: str, path: str | Path) -> SpeechSourceRecord:
    try:
        parser = PARSER_BY_NAME[parser_name]
    except KeyError as exc:
        raise ValueError(f"Unknown speech filename parser: {parser_name}") from exc
    return parser(path)


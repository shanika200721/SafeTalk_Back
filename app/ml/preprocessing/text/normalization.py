"""Conservative text normalization for canonical research outputs."""

from __future__ import annotations

from dataclasses import dataclass
from html import unescape
import re
import unicodedata

from app.ml.preprocessing.text.privacy import replace_privacy_identifiers
from app.ml.preprocessing.text.schemas import TextPrivacySummary

_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")
_LINEBREAK_RE = re.compile(r"\r\n?|\u2028|\u2029")


@dataclass(frozen=True)
class NormalizedText:
    display_text: str
    comparison_text: str
    model_text: str
    privacy_summary: TextPrivacySummary


def normalize_line_breaks(text: str) -> str:
    return _LINEBREAK_RE.sub("\n", text)


def normalize_text(value: object, *, html_unescape: bool = True) -> NormalizedText:
    raw = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", raw)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    if html_unescape:
        text = unescape(text)
    text = normalize_line_breaks(text)
    text, privacy_summary = replace_privacy_identifiers(text)
    text = "\n".join(_WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    comparison = re.sub(r"\s+", " ", text.casefold()).strip()
    return NormalizedText(display_text=text, comparison_text=comparison, model_text=text, privacy_summary=privacy_summary)


def text_contains_only_placeholders(text: str) -> bool:
    cleaned = re.sub(r"\s+", "", text)
    if not cleaned:
        return False
    remaining = cleaned
    for token in ("<URL>", "<EMAIL>", "<PHONE>", "<USER>", "<IP>", "<COMMUNITY>"):
        remaining = remaining.replace(token, "")
    return remaining == ""

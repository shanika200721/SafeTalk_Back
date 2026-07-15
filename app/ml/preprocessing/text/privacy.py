"""Conservative deterministic privacy replacement for text preprocessing."""

from __future__ import annotations

import re

from app.ml.preprocessing.text.schemas import TextPrivacySummary

URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>()]+")
EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
IP_RE = re.compile(r"(?<!\d)(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?!\d)")
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
USERNAME_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{3,}(?![A-Za-z0-9_])")
COMMUNITY_RE = re.compile(r"(?i)(?<!\w)(?:r|u)/[A-Za-z0-9_][A-Za-z0-9_-]{2,}\b")


def _replace(pattern: re.Pattern[str], token: str, text: str) -> tuple[str, int]:
    return pattern.subn(token, text)


def _replace_phone(text: str) -> tuple[str, int]:
    count = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal count
        value = match.group(0)
        digits = re.sub(r"\D", "", value)
        if value.strip().startswith("+") or len(digits) >= 10:
            count += 1
            return "<PHONE>"
        return value

    return PHONE_RE.sub(repl, text), count


def replace_privacy_identifiers(text: str) -> tuple[str, TextPrivacySummary]:
    """Replace identifier-like patterns without storing matched values."""
    safe = str(text)
    safe, url_count = _replace(URL_RE, "<URL>", safe)
    safe, email_count = _replace(EMAIL_RE, "<EMAIL>", safe)
    safe, ip_count = _replace(IP_RE, "<IP>", safe)
    safe, community_count = _replace(COMMUNITY_RE, "<COMMUNITY>", safe)
    safe, username_count = _replace(USERNAME_RE, "<USER>", safe)
    safe, phone_count = _replace_phone(safe)
    summary = TextPrivacySummary(
        url_count=url_count,
        email_count=email_count,
        phone_count=phone_count,
        username_count=username_count,
        ip_address_count=ip_count,
        community_count=community_count,
        possible_person_identifier_count=0,
    )
    return safe, summary

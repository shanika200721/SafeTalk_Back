"""Pseudonymous participant and pilot record identifiers."""

from __future__ import annotations

import hashlib
import re
import secrets
from typing import Iterable, Optional

PARTICIPANT_RE = re.compile(r"^PILOT-P-[A-F0-9]{16}$")
SESSION_RE = re.compile(r"^PILOT-S-[A-F0-9]{16}$")
RECORD_RE = re.compile(r"^PILOT-R-[A-F0-9]{16}$")
PRODUCTION_ID_PATTERNS = (
    re.compile(r"^\d+$"),
    re.compile(r"^user[_-]?\d+$", re.IGNORECASE),
    re.compile(r".+@.+\..+"),
    re.compile(r"^[A-Z]{2,}\d{4,}$"),
)


def _digest(parts: Iterable[object], length: int = 16) -> str:
    joined = "|".join(str(part) for part in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length].upper()


def generate_pilot_participant_id(seed: Optional[int] = None, ordinal: Optional[int] = None) -> str:
    """Return a pseudonymous participant ID.

    Supplying both ``seed`` and ``ordinal`` is deterministic for synthetic
    protocol simulation. Otherwise a non-deterministic token is used.
    """
    token = _digest(("participant", seed, ordinal)) if seed is not None and ordinal is not None else secrets.token_hex(8).upper()
    return f"PILOT-P-{token}"


def validate_pilot_participant_id(value: str) -> bool:
    return bool(PARTICIPANT_RE.match(value or ""))


def generate_session_id(pilot_participant_id: str, session_index: int, seed: Optional[int] = None) -> str:
    validate_no_production_user_id_leakage([pilot_participant_id])
    return f"PILOT-S-{_digest(('session', seed, pilot_participant_id, session_index))}"


def generate_modality_record_id(pilot_participant_id: str, session_id: str, modality: str, index: int = 0) -> str:
    validate_no_production_user_id_leakage([pilot_participant_id, session_id])
    return f"PILOT-R-{_digest(('record', pilot_participant_id, session_id, modality, index))}"


def create_linkage_key(pilot_participant_id: str, protocol_salt: str) -> str:
    """Create a non-reversible key for a separately governed linkage file."""
    if not validate_pilot_participant_id(pilot_participant_id):
        raise ValueError("invalid pilot participant ID")
    if not protocol_salt or len(protocol_salt) < 12:
        raise ValueError("protocol_salt must be non-empty and at least 12 characters")
    return f"LINK-{_digest(('linkage', protocol_salt, pilot_participant_id), length=32)}"


def validate_no_production_user_id_leakage(values: Iterable[object]) -> None:
    for raw in values:
        value = str(raw or "").strip()
        if not value:
            continue
        if value.startswith(("PILOT-P-", "PILOT-S-", "PILOT-R-", "PILOT-E-", "PILOT-O-")):
            continue
        if any(pattern.match(value) for pattern in PRODUCTION_ID_PATTERNS):
            raise ValueError(f"possible production identifier leaked into pilot data: {value}")

import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ml.common import paths
from app.ml.pilot.consent import ConsentDecision, build_consent_record, validate_modality_consent
from app.ml.pilot.constants import (
    PILOT_CONSENT_VERSION,
    PILOT_DATA_SCHEMA_VERSION,
    PILOT_PROTOCOL_VERSION,
    PILOT_RETENTION_POLICY_VERSION,
    PILOT_SAFETY_POLICY_VERSION,
)
from app.ml.pilot.modalities import validate_modality_scope
from app.ml.pilot.participant import (
    create_linkage_key,
    generate_modality_record_id,
    generate_pilot_participant_id,
    generate_session_id,
    validate_no_production_user_id_leakage,
    validate_pilot_participant_id,
)
from app.ml.pilot.privacy import validate_no_direct_identifiers, validate_privacy
from app.ml.pilot.retention import validate_retention_policy
from app.ml.pilot.safety import validate_safety_events
from app.ml.pilot.schemas import (
    PilotModalityRecord,
    PilotOutcomeRecord,
    PilotParticipant,
    PilotSafetyEvent,
    PilotSession,
    PilotWithdrawalRecord,
)
from app.ml.pilot.sessions import (
    align_modality_records,
    calculate_data_completeness,
    detect_duplicate_session_records,
    detect_future_leakage,
)
from app.ml.pilot.synthetic import generate_synthetic_pilot_dataset
from app.ml.pilot.validation import validate_pilot_dataset, validate_protocol_configs


def _load_config(name: str) -> dict:
    return json.loads((paths.get_ml_research_root() / "configs" / name).read_text(encoding="utf-8"))


def _decisions(*, speech=True, face=False, behavioral=False, mood=True, text=True):
    return [
        ConsentDecision("dass21", True, False),
        ConsentDecision("profile", True, False),
        ConsentDecision("mood", mood, False),
        ConsentDecision("text", text, False),
        ConsentDecision("speech", speech, True),
        ConsentDecision("face", face, True),
        ConsentDecision("behavioral", behavioral, True),
    ]


def test_protocol_versions_and_config_schema_only_validation():
    protocol = _load_config("pilot.protocol.v1.json")
    scope = _load_config("pilot.modality_scope.v1.json")
    alignment = _load_config("pilot.alignment_policy.v1.json")
    retention = _load_config("pilot.retention_policy.v1.json")
    assert protocol["protocol_version"] == PILOT_PROTOCOL_VERSION
    assert protocol["schema_version"] == PILOT_DATA_SCHEMA_VERSION
    assert protocol["consent_version"] == PILOT_CONSENT_VERSION
    assert protocol["safety_policy_version"] == PILOT_SAFETY_POLICY_VERSION
    assert retention["retention_policy_version"] == PILOT_RETENTION_POLICY_VERSION
    assert validate_protocol_configs(protocol, scope, alignment, retention, strict=True)["valid"]


def test_valid_participant_invalid_identifier_and_linkage_key():
    pid = generate_pilot_participant_id(seed=7, ordinal=1)
    participant = PilotParticipant(
        pilot_participant_id=pid,
        enrollment_status="enrolled",
        consent_status="active",
        consent_version=PILOT_CONSENT_VERSION,
        enrollment_date=datetime.now(timezone.utc),
        age_band="18-24",
        created_at=datetime.now(timezone.utc),
    )
    assert participant.pilot_participant_id == pid
    assert validate_pilot_participant_id(pid)
    assert not validate_pilot_participant_id("123")
    assert create_linkage_key(pid, "synthetic-salt-123").startswith("LINK-")
    with pytest.raises(ValueError):
        validate_no_production_user_id_leakage(["student@example.edu", "123"])


def test_consent_separate_voice_face_behavioral_decline_and_version_handling():
    now = datetime.now(timezone.utc)
    pid = generate_pilot_participant_id(seed=8, ordinal=1)
    consent = build_consent_record(
        pid,
        _decisions(speech=True, face=False, behavioral=False),
        now,
        "retain_deidentified_derived_destroy_raw_on_withdrawal",
        future_research_permission=False,
        contact_permission=True,
        reviewer_alias="reviewer_a",
    )
    assert "speech" in consent.consented_modalities
    assert "face" in consent.declined_modalities
    assert "behavioral" in consent.declined_modalities
    assert validate_modality_consent(consent, "speech", now + timedelta(minutes=1))
    assert not validate_modality_consent(consent, "face", now + timedelta(minutes=1))
    with pytest.raises(ValueError):
        build_consent_record(pid, _decisions(speech=True)[:-1], now, "retain", False, False, "reviewer")
    with pytest.raises(ValueError):
        build_consent_record(pid, _decisions(speech=True), now, "retain", False, False, "reviewer", consent_version="0.9.0")
    with pytest.raises(ValueError):
        build_consent_record(
            pid,
            [ConsentDecision("speech", True, False), *_decisions(speech=True)[:-1]],
            now,
            "retain",
            False,
            False,
            "reviewer",
        )


def test_timestamp_validation_consent_missing_modality_mismatch_and_withdrawal():
    dataset = generate_synthetic_pilot_dataset()
    scope = _load_config("pilot.modality_scope.v1.json")
    alignment = _load_config("pilot.alignment_policy.v1.json")
    retention = _load_config("pilot.retention_policy.v1.json")
    assert validate_pilot_dataset(dataset.participants, [], dataset.sessions, dataset.modality_records[:1], [], [], [], scope, alignment, retention)["valid"] is False
    bad_record = dataset.modality_records[0]
    bad_record = PilotModalityRecord(
        **{**bad_record.__dict__, "modality": "speech", "record_id": generate_modality_record_id(bad_record.pilot_participant_id, bad_record.session_id, "speech", 99)}
    )
    assert validate_pilot_dataset(dataset.participants, dataset.consents, dataset.sessions, [bad_record], [], [], [], scope, alignment, retention)["valid"] is False
    participant = dataset.participants[0]
    session = dataset.sessions[0]
    withdrawal = PilotWithdrawalRecord(participant.pilot_participant_id, session.scheduled_at, "all_future_collection", True, True, False, None, session.scheduled_at)
    assert validate_pilot_dataset(dataset.participants[:1], dataset.consents[:1], [session], [dataset.modality_records[0]], [], [], [withdrawal], scope, alignment, retention)["valid"] is False
    with pytest.raises(ValueError):
        PilotParticipant("PILOT-P-0000000000000000", "enrolled", "active", PILOT_CONSENT_VERSION, datetime(2026, 1, 1))


def test_alignment_future_leakage_duplicates_and_missingness():
    dataset = generate_synthetic_pilot_dataset()
    alignment = align_modality_records(dataset.modality_records, dataset.sessions, tolerance_minutes=120)
    assert alignment["valid"]
    completeness = calculate_data_completeness(dataset.modality_records, dataset.sessions)
    assert completeness["mood"]["missingness"] > 0
    duplicate = detect_duplicate_session_records([dataset.modality_records[0], dataset.modality_records[0]])
    assert not duplicate["valid"]
    first = dataset.modality_records[0]
    future_outcome = PilotOutcomeRecord(
        "PILOT-O-0000000000000000",
        first.pilot_participant_id,
        first.collected_at - timedelta(minutes=1),
        "counselor_reviewed_follow_up",
        True,
        True,
        "counselor_concern_level_only",
        "synthetic_low",
    )
    assert not detect_future_leakage([first], [future_outcome])["valid"]


def test_privacy_rejects_production_ids_email_names_and_bad_biometric_paths():
    dataset = generate_synthetic_pilot_dataset()
    assert validate_privacy(dataset.modality_records, [p.pilot_participant_id for p in dataset.participants])["valid"]
    with pytest.raises(ValueError):
        validate_no_direct_identifiers({"email": "student@example.edu"})
    with pytest.raises(ValueError):
        validate_no_direct_identifiers({"full_name": "Example Student"})
    speech = next(record for record in dataset.modality_records if record.modality == "speech")
    bad = PilotModalityRecord(**{**speech.__dict__, "raw_artifact_reference": "outside/speech.wav"})
    assert not validate_privacy([bad], [bad.pilot_participant_id])["valid"]


def test_safety_human_escalation_no_model_decision_and_blinded_outcomes():
    now = datetime.now(timezone.utc)
    event = PilotSafetyEvent("PILOT-E-0000000000000000", "PILOT-P-0000000000000000", "questionnaire_response", "self_disclosure", "questionnaire_response", "refer to counselor", True, "open", now)
    assert validate_safety_events([event])["valid"]
    bad = PilotSafetyEvent("PILOT-E-0000000000000001", "PILOT-P-0000000000000000", "model_prediction", "risk", "researcher", "", False, "open", now)
    assert not validate_safety_events([bad])["valid"]
    outcome = PilotOutcomeRecord("PILOT-O-0000000000000001", "PILOT-P-0000000000000000", now, "model_output", True, False, "suicide-risk truth", "high")
    dataset = generate_synthetic_pilot_dataset()
    scope = _load_config("pilot.modality_scope.v1.json")
    alignment = _load_config("pilot.alignment_policy.v1.json")
    retention = _load_config("pilot.retention_policy.v1.json")
    assert not validate_pilot_dataset(dataset.participants, dataset.consents, dataset.sessions, dataset.modality_records, [outcome], [], [], scope, alignment, retention)["valid"]


def test_modality_scope_retention_and_real_collection_disabled():
    scope = _load_config("pilot.modality_scope.v1.json")
    retention = _load_config("pilot.retention_policy.v1.json")
    result = validate_modality_scope(scope)
    assert result["valid"]
    enabled_real = [row["modality"] for row in result["matrix"] if row["enabled_for_real_collection"]]
    assert enabled_real == []
    assert validate_retention_policy(retention)["valid"]
    bad_retention = json.loads(json.dumps(retention))
    bad_retention["categories"]["model_outputs"]["retention_period"] = "retain"
    assert not validate_retention_policy(bad_retention)["valid"]


def test_synthetic_simulation_deterministic_counts_mixed_consent_and_no_predictions():
    first = generate_synthetic_pilot_dataset(participants=12, weeks=4, seed=404)
    second = generate_synthetic_pilot_dataset(participants=12, weeks=4, seed=404)
    assert [p.pilot_participant_id for p in first.participants] == [p.pilot_participant_id for p in second.participants]
    assert len(first.participants) == 12
    assert len(first.sessions) == 382
    assert len(first.modality_records) == 267
    assert len(first.withdrawals) == 1
    assert len(first.safety_events) == 1
    assert any("speech" in consent.consented_modalities for consent in first.consents)
    assert any("speech" in consent.declined_modalities for consent in first.consents)
    assert all(outcome.outcome_label != "suicide-risk truth" for outcome in first.outcomes)
    assert first.metadata["model_execution"] is False
    assert first.metadata["prediction_generation"] is False
    assert first.metadata["alert_generation"] is False


def test_cli_schema_only_synthetic_smoke_overwrite_refusal_and_no_side_effect_terms():
    script = paths.get_backend_root() / "scripts" / "validate_pilot_protocol.py"
    suffix = uuid.uuid4().hex
    smoke = f"generated/temporary/phase4a-smoke-{suffix}"
    report = f"generated/temporary/phase4a-readiness-{suffix}"
    schema = subprocess.run(
        [sys.executable, str(script), "--schema-only", "--report-path", report, "--overwrite"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert schema.returncode == 0
    normal = subprocess.run(
        [sys.executable, str(script), "--synthetic-smoke", "--output-dir", smoke, "--report-path", report, "--participants", "12", "--weeks", "4", "--seed", "404", "--overwrite"],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert normal.returncode == 0
    assert "participants=12" in normal.stdout
    refusal = subprocess.run(
        [sys.executable, str(script), "--synthetic-smoke", "--output-dir", smoke, "--report-path", report],
        cwd=paths.get_backend_root(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert refusal.returncode != 0
    summary = json.loads((paths.get_repository_root() / report / "pilot_readiness_summary.json").read_text(encoding="utf-8"))
    assert summary["real_collection_prohibited"]
    source = script.read_text(encoding="utf-8")
    assert "SessionLocal" not in source
    assert "ModalityPrediction" not in source
    assert "RiskAssessment" not in source
    assert "Alert(" not in source

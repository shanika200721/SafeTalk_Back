import sys
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


ML_TEST_FILES = {
    "test_fusion_controlled_v2.py",
    "test_phase2_behavioral_preprocessing.py",
    "test_phase2_config_schemas.py",
    "test_phase2_dass21_scoring.py",
    "test_phase2_dataset_audit.py",
    "test_phase2_dataset_fingerprinting.py",
    "test_phase2_face_preprocessing.py",
    "test_phase2_mood_preprocessing.py",
    "test_phase2_profile_preprocessing.py",
    "test_phase2_speech_preprocessing.py",
    "test_phase2_text_preprocessing.py",
    "test_phase3a_split_design.py",
    "test_phase3b_training_framework.py",
    "test_phase3c_profile_baseline.py",
    "test_phase3d_text_baseline.py",
    "test_phase3e_speech_baseline.py",
    "test_phase3f_speech_domain_shift.py",
    "test_phase3g_face_remediation.py",
    "test_phase3h_face_human_review.py",
    "test_phase3i_face_baseline.py",
    "test_phase3j_model_governance.py",
    "test_phase3k_synthetic_fusion.py",
    "test_phase4d_model_governance.py",
    "test_phase4e_controlled_fusion.py",
    "test_phase4p_seven_modalities.py",
}

DATASET_VALIDATION_TEST_FILES = {
    "test_phase2_cross_modality_validation.py",
    "test_phase2_dataset_audit.py",
    "test_phase2_dataset_fingerprinting.py",
    "test_phase2_face_preprocessing.py",
    "test_phase2_mood_preprocessing.py",
    "test_phase2_profile_preprocessing.py",
    "test_phase2_speech_preprocessing.py",
    "test_phase2_text_preprocessing.py",
    "test_phase3a_split_design.py",
}

RUNTIME_TEST_FILES = {
    "test_phase4k_runtime_activation.py",
    "test_phase4l_profile_camera.py",
    "test_phase4m_counselor_speech_integration.py",
    "test_phase4n_runtime_completion.py",
    "test_phase4o_speech_runtime.py",
    "test_phase4r_runtime_hardening.py",
}

INTEGRATION_TEST_FILES = {
    "test_phase1_database_foundation.py",
    "test_phase4b_auth_consent.py",
    "test_phase4c_modality_contracts.py",
    "test_phase4d_model_governance.py",
    "test_phase4e_controlled_fusion.py",
    "test_phase4g_counselor_workflow.py",
    "test_phase4g_support_contacts.py",
    "test_phase4h_admin_portal.py",
    "test_phase4i_student_wellness.py",
    "test_phase4k_runtime_activation.py",
    "test_phase4l_profile_camera.py",
    "test_phase4m_counselor_speech_integration.py",
    "test_phase4n_runtime_completion.py",
    "test_phase4p_seven_modalities.py",
    "test_phase4q_validation.py",
    "test_phase4r_runtime_hardening.py",
}

SLOW_TEST_FILES = {
    "test_phase2_cross_modality_validation.py",
    "test_phase2_face_preprocessing.py",
    "test_phase2_mood_preprocessing.py",
    "test_phase3a_split_design.py",
    "test_phase3e_speech_baseline.py",
    "test_phase3f_speech_domain_shift.py",
    "test_phase3g_face_remediation.py",
    "test_phase3h_face_human_review.py",
    "test_phase3i_face_baseline.py",
    "test_phase3k_synthetic_fusion.py",
}

E2E_TEST_FILES = {
    "test_phase4q_validation.py",
}

KNOWN_MARKERS = {"unit", "integration", "runtime", "ml", "dataset_validation", "slow", "e2e"}

SLOW_TEST_NAMES = {
    ("test_phase4a_pilot_protocol.py", "test_cli_schema_only_synthetic_smoke_overwrite_refusal_and_no_side_effect_terms"),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        filename = Path(str(item.fspath)).name
        markers = set()
        if filename in ML_TEST_FILES:
            markers.add("ml")
        if filename in DATASET_VALIDATION_TEST_FILES:
            markers.add("dataset_validation")
        if filename in RUNTIME_TEST_FILES:
            markers.add("runtime")
        if filename in INTEGRATION_TEST_FILES:
            markers.add("integration")
        if filename in SLOW_TEST_FILES:
            markers.add("slow")
        if (filename, item.name) in SLOW_TEST_NAMES:
            markers.add("slow")
        if filename in E2E_TEST_FILES:
            markers.add("e2e")

        existing = {mark.name for mark in item.iter_markers()}
        for marker in sorted(markers - existing):
            item.add_marker(getattr(pytest.mark, marker))

        if not (existing | markers) & KNOWN_MARKERS:
            item.add_marker(pytest.mark.unit)

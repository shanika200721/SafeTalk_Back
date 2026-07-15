"""Privacy, fairness, and governance limitation synthesis."""

from __future__ import annotations

from typing import Dict, List


LIMITATIONS: Dict[str, List[Dict[str, str]]] = {
    "profile": [
        {"code": "PROFILE_SENSITIVE_EXCLUDED", "limitation": "Sensitive attributes excluded from primary model.", "recommendation": "Do not infer fairness from this baseline."},
        {"code": "PROFILE_FAIRNESS_SMALL_N", "limitation": "Fairness is not meaningfully measurable due small sample.", "recommendation": "Collect consented data large enough for subgroup analysis."},
    ],
    "text": [
        {"code": "TEXT_INCOMPLETE_ANON", "limitation": "Text anonymization/privacy normalization is incomplete.", "recommendation": "Strengthen de-identification and audit residual identifiers."},
        {"code": "TEXT_SOCIAL_MEDIA_DOMAIN", "limitation": "Social-media domain may not reflect local counseling or student text.", "recommendation": "Externally validate on approved target-domain text."},
        {"code": "TEXT_AUTHOR_GROUPING", "limitation": "Author grouping is incomplete.", "recommendation": "Use participant-level grouping before future training."},
    ],
    "speech": [
        {"code": "SPEECH_BIOMETRIC", "limitation": "Voice data is biometric and sensitive.", "recommendation": "Require explicit consent and secure handling."},
        {"code": "SPEECH_CORPUS_BIAS", "limitation": "Speaker/corpus bias and shortcut risk are strong.", "recommendation": "Use natural local speech and corpus-held-out validation."},
        {"code": "SPEECH_ACCENT_DEVICE", "limitation": "Accent and device differences are unvalidated.", "recommendation": "Evaluate across local accents and collection devices."},
    ],
    "face": [
        {"code": "FACE_BIOMETRIC", "limitation": "Face data is biometric and highly sensitive.", "recommendation": "Do not deploy without explicit biometric governance approval."},
        {"code": "FACE_DEMOGRAPHICS_UNAVAILABLE", "limitation": "Demographics unavailable; fairness cannot be established.", "recommendation": "Collect governed metadata only if ethically approved."},
        {"code": "FACE_CAPTURE_CONDITIONS", "limitation": "Skin tone, lighting, pose, and camera variation are unvalidated.", "recommendation": "Perform controlled external validation before any future use."},
        {"code": "FACE_REVIEW_INDEPENDENCE", "limitation": "Reviewer independence is unverified.", "recommendation": "Require independent review protocol before new training."},
    ],
    "mood": [
        {"code": "MOOD_LONGITUDINAL_CONSENT", "limitation": "Longitudinal mood tracking raises consent and secondary-use concerns.", "recommendation": "Define opt-in protocol and retention limits before collection."},
    ],
    "behavioral": [
        {"code": "BEHAVIORAL_COVERT_MONITORING", "limitation": "Behavioral data can become covert monitoring.", "recommendation": "Require explicit consent and narrow purpose limitation."},
        {"code": "BEHAVIORAL_NO_REAL_DATA", "limitation": "No real source data exists.", "recommendation": "Do not train until approved real data exists."},
    ],
}


def build_privacy_fairness_limitations() -> Dict[str, List[Dict[str, str]]]:
    return {modality: list(records) for modality, records in LIMITATIONS.items()}

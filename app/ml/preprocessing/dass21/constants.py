"""Authoritative DASS-21 scoring constants.

The application stores DASS-21 answers as ordered 0-3 responses. The audited
research source is DASS-42 and uses 1-4 response columns; dataset conversion
lives in dataset_mapping.py so scoring does not guess source conventions.
"""

from __future__ import annotations

DASS21_SCORING_VERSION = "1.0.0"
DASS21_FEATURE_SCHEMA_VERSION = "1.0.0"
DASS21_ITEM_MAPPING_VERSION = "1.0.0"

QUESTIONNAIRE_VERSION = "DASS-21"
RESPONSE_SCALE_0_3 = "0-3"
RESPONSE_MIN = 0
RESPONSE_MAX = 3
ITEM_MULTIPLIER = 2
SUBSCALE_MAX_RAW_SCORE = 21
SUBSCALE_MAX_MULTIPLIED_SCORE = 42

SUBSCALES = ("depression", "anxiety", "stress")
SEVERITY_LEVELS = ("normal", "mild", "moderate", "severe", "extremely_severe")

DASS21_EXPECTED_ITEMS = tuple(f"Q{index}" for index in range(1, 22))

# Canonical DASS-21 item positions in the 21-question application payload.
DASS21_ITEM_MAPPING = {
    "depression": ("Q3", "Q5", "Q10", "Q13", "Q16", "Q17", "Q21"),
    "anxiety": ("Q2", "Q4", "Q7", "Q9", "Q15", "Q19", "Q20"),
    "stress": ("Q1", "Q6", "Q8", "Q11", "Q12", "Q14", "Q18"),
}

# Final-score thresholds after the DASS-21 raw subscale score is multiplied by 2.
DASS21_SEVERITY_THRESHOLDS = {
    "depression": {
        "normal": (0, 9),
        "mild": (10, 13),
        "moderate": (14, 20),
        "severe": (21, 27),
        "extremely_severe": (28, 42),
    },
    "anxiety": {
        "normal": (0, 7),
        "mild": (8, 9),
        "moderate": (10, 14),
        "severe": (15, 19),
        "extremely_severe": (20, 42),
    },
    "stress": {
        "normal": (0, 14),
        "mild": (15, 18),
        "moderate": (19, 25),
        "severe": (26, 33),
        "extremely_severe": (34, 42),
    },
}

# Official DASS-21 short-form subset when selected from the audited DASS-42 source.
DASS42_TO_DASS21_SOURCE_COLUMNS = {
    "Q1": "Q22A",
    "Q2": "Q2A",
    "Q3": "Q3A",
    "Q4": "Q4A",
    "Q5": "Q42A",
    "Q6": "Q6A",
    "Q7": "Q41A",
    "Q8": "Q12A",
    "Q9": "Q40A",
    "Q10": "Q10A",
    "Q11": "Q39A",
    "Q12": "Q8A",
    "Q13": "Q26A",
    "Q14": "Q35A",
    "Q15": "Q28A",
    "Q16": "Q31A",
    "Q17": "Q17A",
    "Q18": "Q18A",
    "Q19": "Q25A",
    "Q20": "Q20A",
    "Q21": "Q21A",
}

TIMING_COLUMN_PATTERN = r"^Q\d+E$"
POSITION_COLUMN_PATTERN = r"^Q\d+I$"
RESPONSE_COLUMN_PATTERN = r"^Q\d+A$"

DEMOGRAPHIC_COLUMNS = (
    "country",
    "education",
    "urban",
    "gender",
    "engnat",
    "age",
    "hand",
    "religion",
    "orientation",
    "race",
    "voted",
    "married",
    "familysize",
    "major",
)

METADATA_COLUMNS = (
    "introelapse",
    "source",
    "screensize",
    "surveyelapse",
    "testelapse",
    "uniquenetworklocation",
)


def get_threshold_metadata() -> dict:
    """Return threshold metadata for API and documentation use."""

    return {
        "questionnaire_version": QUESTIONNAIRE_VERSION,
        "scoring_version": DASS21_SCORING_VERSION,
        "response_scale": RESPONSE_SCALE_0_3,
        "raw_subscale_max": SUBSCALE_MAX_RAW_SCORE,
        "multiplied_subscale_max": SUBSCALE_MAX_MULTIPLIED_SCORE,
        "multiplier": ITEM_MULTIPLIER,
        "severity_levels": list(SEVERITY_LEVELS),
        "thresholds": {
            subscale: {
                severity: {"minimum": minimum, "maximum": maximum}
                for severity, (minimum, maximum) in thresholds.items()
            }
            for subscale, thresholds in DASS21_SEVERITY_THRESHOLDS.items()
        },
    }

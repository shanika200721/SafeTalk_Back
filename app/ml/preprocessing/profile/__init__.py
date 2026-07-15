"""Student Profile preprocessing foundation."""

from app.ml.preprocessing.profile.constants import (
    PROFILE_FEATURE_SCHEMA_VERSION,
    PROFILE_MAPPING_VERSION,
    PROFILE_PREPROCESSING_VERSION,
)
from app.ml.preprocessing.profile.preprocessor import preprocess_profile_dataset

__all__ = [
    "PROFILE_FEATURE_SCHEMA_VERSION",
    "PROFILE_MAPPING_VERSION",
    "PROFILE_PREPROCESSING_VERSION",
    "preprocess_profile_dataset",
]

"""phase 4l profile assessment and camera entry metadata

Revision ID: 20260725_0011
Revises: 20260725_0010
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0011"
down_revision = "20260725_0010"
branch_labels = None
depends_on = None


CONSENT_CHECK = (
    "consent_type IN ('profile_processing', 'profile_data_storage', 'profile_model_processing', "
    "'dass21_processing', 'mood_processing', 'text_processing', 'voice_processing', "
    "'face_processing', 'behavioral_processing', 'facial_capture', 'facial_model_processing', "
    "'counselor_escalation', 'research_data_use')"
)

OLD_CONSENT_CHECK = (
    "consent_type IN ('profile_processing', 'dass21_processing', 'mood_processing', "
    "'text_processing', 'voice_processing', 'face_processing', 'behavioral_processing', "
    "'counselor_escalation', 'research_data_use')"
)


def upgrade() -> None:
    with op.batch_alter_table("profile_assessments") as batch_op:
        batch_op.add_column(sa.Column("profile_assessment_id", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("questionnaire_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("responses_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("normalized_features_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("preprocessing_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("started_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("submitted_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("completed_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("stale_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("prediction_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("consent_record_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("source", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("privacy_metadata_json", sa.JSON(), nullable=True))
        batch_op.create_foreign_key("fk_profile_assessments_prediction", "modality_predictions", ["prediction_id"], ["id"])
        batch_op.create_foreign_key("fk_profile_assessments_consent_record", "consent_records", ["consent_record_id"], ["id"])
        batch_op.create_index("ix_profile_assessments_profile_assessment_id", ["profile_assessment_id"], unique=True)
        batch_op.create_index("ix_profile_assessments_status", ["status"])

    with op.batch_alter_table("consent_records") as batch_op:
        batch_op.drop_constraint("ck_consent_records_known_type", type_="check")
        batch_op.create_check_constraint("ck_consent_records_known_type", CONSENT_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("consent_records") as batch_op:
        batch_op.drop_constraint("ck_consent_records_known_type", type_="check")
        batch_op.create_check_constraint("ck_consent_records_known_type", OLD_CONSENT_CHECK)

    with op.batch_alter_table("profile_assessments") as batch_op:
        batch_op.drop_index("ix_profile_assessments_status")
        batch_op.drop_index("ix_profile_assessments_profile_assessment_id")
        batch_op.drop_constraint("fk_profile_assessments_consent_record", type_="foreignkey")
        batch_op.drop_constraint("fk_profile_assessments_prediction", type_="foreignkey")
        batch_op.drop_column("privacy_metadata_json")
        batch_op.drop_column("source")
        batch_op.drop_column("consent_record_id")
        batch_op.drop_column("prediction_id")
        batch_op.drop_column("stale_at")
        batch_op.drop_column("completed_at")
        batch_op.drop_column("submitted_at")
        batch_op.drop_column("started_at")
        batch_op.drop_column("preprocessing_version")
        batch_op.drop_column("normalized_features_json")
        batch_op.drop_column("responses_json")
        batch_op.drop_column("status")
        batch_op.drop_column("questionnaire_version")
        batch_op.drop_column("profile_assessment_id")

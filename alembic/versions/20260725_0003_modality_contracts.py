"""add canonical modality contracts

Revision ID: 20260725_0003
Revises: 20260725_0002
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0003"
down_revision = "20260725_0002"
branch_labels = None
depends_on = None


MODALITY_CHECK = "modality IN ('profile', 'dass21', 'mood', 'text', 'speech', 'face', 'behavioral')"
STATUS_CHECK = "status IN ('pending', 'succeeded', 'failed', 'unavailable', 'rejected', 'stale')"
OUTPUT_TYPE_CHECK = "output_type IN ('rule_based', 'heuristic', 'machine_learning', 'manual', 'externally_supplied')"


def upgrade() -> None:
    op.add_column("dass21_assessments", sa.Column("questionnaire_version", sa.String(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("item_mapping_version", sa.String(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("scoring_version", sa.String(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("score_multiplier", sa.Float(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("completed_item_count", sa.Integer(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("is_complete", sa.Boolean(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("scored_at", sa.DateTime(), nullable=True))
    op.add_column("dass21_assessments", sa.Column("consent_policy_version", sa.String(), nullable=True))
    op.execute(
        """
        UPDATE dass21_assessments
        SET questionnaire_version = COALESCE(questionnaire_version, 'DASS-21'),
            item_mapping_version = COALESCE(item_mapping_version, '1.0.0'),
            scoring_version = COALESCE(scoring_version, '1.0.0'),
            score_multiplier = COALESCE(score_multiplier, 2.0),
            completed_item_count = COALESCE(completed_item_count, 21),
            is_complete = COALESCE(is_complete, TRUE),
            scored_at = COALESCE(scored_at, created_at),
            consent_policy_version = COALESCE(consent_policy_version, '1.0')
        """
    )

    with op.batch_alter_table("feature_snapshots") as batch_op:
        batch_op.alter_column("source_record_id", existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(sa.Column("source_timestamp", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("feature_schema_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("consent_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("data_quality_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("data_quality_flags", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))

    op.create_index(
        "ix_feature_snapshots_student_modality_source",
        "feature_snapshots",
        ["student_id", "modality", "source_type", "source_record_id"],
        unique=False,
    )

    with op.batch_alter_table("modality_predictions") as batch_op:
        batch_op.alter_column("source_record_id", existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column("model_registry_id", existing_type=sa.Integer(), nullable=True)
        batch_op.alter_column("predicted_class", existing_type=sa.String(), nullable=True)
        batch_op.alter_column("probability", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("score_0_100", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("confidence", existing_type=sa.Float(), nullable=True)
        batch_op.add_column(sa.Column("feature_snapshot_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("status", sa.String(), nullable=False, server_default="succeeded"))
        batch_op.add_column(sa.Column("is_available", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("failure_code", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("failure_message_safe", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_timestamp", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("generated_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("valid_until", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("model_name", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("model_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("preprocessing_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("feature_schema_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("output_type", sa.String(), nullable=False, server_default="machine_learning"))
        batch_op.add_column(sa.Column("label", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("raw_output_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("consent_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("evidence_available", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("clinical_use_boundary", sa.String(), nullable=False, server_default="screening_support_only"))
        batch_op.add_column(sa.Column("data_quality_status", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("data_quality_flags", sa.JSON(), nullable=True))
        batch_op.create_foreign_key(
            "fk_modality_predictions_feature_snapshot_id_feature_snapshots",
            "feature_snapshots",
            ["feature_snapshot_id"],
            ["id"],
        )
        batch_op.create_check_constraint("ck_modality_predictions_canonical_modality", MODALITY_CHECK)
        batch_op.create_check_constraint("ck_modality_predictions_status", STATUS_CHECK)
        batch_op.create_check_constraint("ck_modality_predictions_output_type", OUTPUT_TYPE_CHECK)

    op.execute("UPDATE modality_predictions SET generated_at = COALESCE(generated_at, created_at)")
    with op.batch_alter_table("modality_predictions") as batch_op:
        batch_op.alter_column("generated_at", existing_type=sa.DateTime(), nullable=False)

    op.create_index(
        "ix_modality_predictions_student_modality_status",
        "modality_predictions",
        ["student_id", "modality", "status"],
        unique=False,
    )


def downgrade() -> None:
    # Downgrade is additive-reversal only. If Phase 4C null-status records exist,
    # export or remove them before restoring old NOT NULL prediction columns.
    op.drop_index("ix_modality_predictions_student_modality_status", table_name="modality_predictions")
    with op.batch_alter_table("modality_predictions") as batch_op:
        batch_op.drop_constraint("ck_modality_predictions_output_type", type_="check")
        batch_op.drop_constraint("ck_modality_predictions_status", type_="check")
        batch_op.drop_constraint("ck_modality_predictions_canonical_modality", type_="check")
        batch_op.drop_constraint("fk_modality_predictions_feature_snapshot_id_feature_snapshots", type_="foreignkey")
        batch_op.drop_column("data_quality_flags")
        batch_op.drop_column("data_quality_status")
        batch_op.drop_column("clinical_use_boundary")
        batch_op.drop_column("evidence_available")
        batch_op.drop_column("consent_policy_version")
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("raw_output_json")
        batch_op.drop_column("label")
        batch_op.drop_column("output_type")
        batch_op.drop_column("feature_schema_version")
        batch_op.drop_column("preprocessing_version")
        batch_op.drop_column("model_version")
        batch_op.drop_column("model_name")
        batch_op.drop_column("valid_until")
        batch_op.drop_column("generated_at")
        batch_op.drop_column("source_timestamp")
        batch_op.drop_column("failure_message_safe")
        batch_op.drop_column("failure_code")
        batch_op.drop_column("is_available")
        batch_op.drop_column("status")
        batch_op.drop_column("feature_snapshot_id")
        batch_op.alter_column("confidence", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("score_0_100", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("probability", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("predicted_class", existing_type=sa.String(), nullable=False)
        batch_op.alter_column("model_registry_id", existing_type=sa.Integer(), nullable=False)
        batch_op.alter_column("source_record_id", existing_type=sa.Integer(), nullable=False)

    op.drop_index("ix_feature_snapshots_student_modality_source", table_name="feature_snapshots")
    with op.batch_alter_table("feature_snapshots") as batch_op:
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("data_quality_flags")
        batch_op.drop_column("data_quality_status")
        batch_op.drop_column("consent_policy_version")
        batch_op.drop_column("feature_schema_version")
        batch_op.drop_column("source_timestamp")
        batch_op.alter_column("source_record_id", existing_type=sa.Integer(), nullable=False)

    op.drop_column("dass21_assessments", "consent_policy_version")
    op.drop_column("dass21_assessments", "scored_at")
    op.drop_column("dass21_assessments", "is_complete")
    op.drop_column("dass21_assessments", "completed_item_count")
    op.drop_column("dass21_assessments", "score_multiplier")
    op.drop_column("dass21_assessments", "scoring_version")
    op.drop_column("dass21_assessments", "item_mapping_version")
    op.drop_column("dass21_assessments", "questionnaire_version")

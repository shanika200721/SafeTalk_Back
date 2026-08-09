"""add controlled fusion runtime audit fields

Revision ID: 20260725_0005
Revises: 20260725_0004
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0005"
down_revision = "20260725_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("risk_assessments") as batch_op:
        batch_op.alter_column("final_probability", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("final_score", existing_type=sa.Float(), nullable=True)
        batch_op.alter_column("risk_level", existing_type=sa.String(), nullable=True)
        batch_op.alter_column("confidence", existing_type=sa.Float(), nullable=True)
        batch_op.add_column(sa.Column("status", sa.String(), nullable=False, server_default="completed"))
        batch_op.add_column(sa.Column("assessment_type", sa.String(), nullable=False, server_default="screening_support"))
        batch_op.add_column(sa.Column("model_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("model_risk_level", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("fusion_config_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("fusion_config_hash", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("threshold_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("mapping_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("staleness_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("coverage_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("configured_modalities", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("available_modalities", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("used_modalities", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("missing_modalities", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("excluded_modalities", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("evidence_coverage", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("coverage_category", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("effective_weights", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("latest_source_timestamp", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("oldest_source_timestamp", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("evidence_window_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("limitations_json", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("screening_only", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("model_output_only", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("human_review_status", sa.String(), nullable=False, server_default="not_requested"))
        batch_op.add_column(sa.Column("counselor_decision", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("counselor_override", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("alert_created", sa.Boolean(), nullable=False, server_default=sa.false()))

    with op.batch_alter_table("risk_assessment_inputs") as batch_op:
        batch_op.alter_column("modality_prediction_id", existing_type=sa.Integer(), nullable=True)
        batch_op.add_column(sa.Column("modality", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("source_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("mapped_score", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("base_weight", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("effective_weight", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("included", sa.Boolean(), nullable=False, server_default=sa.true()))
        batch_op.add_column(sa.Column("exclusion_reason", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("source_timestamp", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("prediction_age_seconds", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("mapping_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))


def downgrade() -> None:
    # Export Phase 4E controlled-fusion records before downgrading if they use
    # insufficient-evidence null scores or per-input audit fields.
    with op.batch_alter_table("risk_assessment_inputs") as batch_op:
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("mapping_version")
        batch_op.drop_column("prediction_age_seconds")
        batch_op.drop_column("source_timestamp")
        batch_op.drop_column("exclusion_reason")
        batch_op.drop_column("included")
        batch_op.drop_column("effective_weight")
        batch_op.drop_column("base_weight")
        batch_op.drop_column("mapped_score")
        batch_op.drop_column("source_score")
        batch_op.drop_column("modality")
        batch_op.alter_column("modality_prediction_id", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("risk_assessments") as batch_op:
        batch_op.drop_column("alert_created")
        batch_op.drop_column("counselor_override")
        batch_op.drop_column("counselor_decision")
        batch_op.drop_column("human_review_status")
        batch_op.drop_column("model_output_only")
        batch_op.drop_column("screening_only")
        batch_op.drop_column("limitations_json")
        batch_op.drop_column("evidence_window_json")
        batch_op.drop_column("oldest_source_timestamp")
        batch_op.drop_column("latest_source_timestamp")
        batch_op.drop_column("effective_weights")
        batch_op.drop_column("coverage_category")
        batch_op.drop_column("evidence_coverage")
        batch_op.drop_column("excluded_modalities")
        batch_op.drop_column("missing_modalities")
        batch_op.drop_column("used_modalities")
        batch_op.drop_column("available_modalities")
        batch_op.drop_column("configured_modalities")
        batch_op.drop_column("coverage_policy_version")
        batch_op.drop_column("staleness_policy_version")
        batch_op.drop_column("mapping_version")
        batch_op.drop_column("threshold_version")
        batch_op.drop_column("fusion_config_hash")
        batch_op.drop_column("fusion_config_version")
        batch_op.drop_column("model_risk_level")
        batch_op.drop_column("model_score")
        batch_op.drop_column("assessment_type")
        batch_op.drop_column("status")
        batch_op.alter_column("confidence", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("risk_level", existing_type=sa.String(), nullable=False)
        batch_op.alter_column("final_score", existing_type=sa.Float(), nullable=False)
        batch_op.alter_column("final_probability", existing_type=sa.Float(), nullable=False)

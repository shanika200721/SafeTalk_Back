"""add model registry governance fields

Revision ID: 20260725_0004
Revises: 20260725_0003
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0004"
down_revision = "20260725_0003"
branch_labels = None
depends_on = None


REGISTRY_STATUS_CHECK = (
    "status IN ('discovered', 'verified', 'active', 'inactive', 'rejected', 'corrupt', 'incompatible')"
)


def upgrade() -> None:
    op.add_column("model_registry", sa.Column("artifact_sha256", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("serializer", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("framework_version", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("preprocessing_version", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("label_mapping_version", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("training_dataset_identifier", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("training_split_identifier", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("evaluation_report_identifier", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("model_card_path", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("status", sa.String(), nullable=False, server_default="discovered"))
    op.add_column("model_registry", sa.Column("verification_status", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("verification_checked_at", sa.DateTime(), nullable=True))
    op.add_column("model_registry", sa.Column("verification_failure_code", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("verification_message", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("verification_json", sa.JSON(), nullable=True))
    op.add_column("model_registry", sa.Column("approved_at", sa.DateTime(), nullable=True))
    op.add_column("model_registry", sa.Column("approved_by", sa.String(), nullable=True))
    op.add_column("model_registry", sa.Column("notes", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("limitations_json", sa.JSON(), nullable=True))
    op.add_column("model_registry", sa.Column("intended_use", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("prohibited_use", sa.Text(), nullable=True))
    op.add_column("model_registry", sa.Column("metadata_json", sa.JSON(), nullable=True))
    op.create_index(op.f("ix_model_registry_status"), "model_registry", ["status"], unique=False)
    op.create_index(
        "uq_model_registry_one_active_modality",
        "model_registry",
        ["modality"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )
    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.create_check_constraint("ck_model_registry_status", REGISTRY_STATUS_CHECK)


def downgrade() -> None:
    op.drop_index("uq_model_registry_one_active_modality", table_name="model_registry")
    with op.batch_alter_table("model_registry") as batch_op:
        batch_op.drop_constraint("ck_model_registry_status", type_="check")
    op.drop_index(op.f("ix_model_registry_status"), table_name="model_registry")
    op.drop_column("model_registry", "metadata_json")
    op.drop_column("model_registry", "prohibited_use")
    op.drop_column("model_registry", "intended_use")
    op.drop_column("model_registry", "limitations_json")
    op.drop_column("model_registry", "notes")
    op.drop_column("model_registry", "approved_by")
    op.drop_column("model_registry", "approved_at")
    op.drop_column("model_registry", "verification_json")
    op.drop_column("model_registry", "verification_message")
    op.drop_column("model_registry", "verification_failure_code")
    op.drop_column("model_registry", "verification_checked_at")
    op.drop_column("model_registry", "verification_status")
    op.drop_column("model_registry", "status")
    op.drop_column("model_registry", "model_card_path")
    op.drop_column("model_registry", "evaluation_report_identifier")
    op.drop_column("model_registry", "training_split_identifier")
    op.drop_column("model_registry", "training_dataset_identifier")
    op.drop_column("model_registry", "label_mapping_version")
    op.drop_column("model_registry", "preprocessing_version")
    op.drop_column("model_registry", "framework_version")
    op.drop_column("model_registry", "serializer")
    op.drop_column("model_registry", "artifact_sha256")

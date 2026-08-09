"""phase 4k runtime activation metadata and journal persistence

Revision ID: 20260725_0010
Revises: 20260725_0009
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0010"
down_revision = "20260725_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("safetalk_bot_messages") as batch_op:
        batch_op.add_column(sa.Column("response_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("response_variant_id", sa.String(), nullable=True))

    with op.batch_alter_table("safetalk_conversations") as batch_op:
        batch_op.add_column(sa.Column("response_policy_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("legacy_response_version", sa.String(), nullable=True))

    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(sa.Column("ai_analysis_requested", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column("ai_analysis_status", sa.String(), nullable=False, server_default="not_requested"))
        batch_op.add_column(sa.Column("ai_prediction_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))
        batch_op.create_foreign_key("fk_chat_messages_ai_prediction", "modality_predictions", ["ai_prediction_id"], ["id"])

    op.create_table(
        "journal_entries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("journal_entry_id", sa.String(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("mood_tag", sa.String(), nullable=True),
        sa.Column("tags_json", sa.JSON(), nullable=True),
        sa.Column("entry_date", sa.DateTime(), nullable=False),
        sa.Column("ai_analysis_opt_in", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("analysis_status", sa.String(), nullable=False, server_default="not_requested"),
        sa.Column("analysis_consent_record_id", sa.Integer(), nullable=True),
        sa.Column("shared_with_counselor", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("shared_at", sa.DateTime(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["analysis_consent_record_id"], ["consent_records.id"]),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("journal_entry_id"),
    )
    op.create_index("ix_journal_entries_id", "journal_entries", ["id"])
    op.create_index("ix_journal_entries_journal_entry_id", "journal_entries", ["journal_entry_id"])
    op.create_index("ix_journal_entries_student_created", "journal_entries", ["student_id", "created_at"])
    op.create_index("ix_journal_entries_student_entry_date", "journal_entries", ["student_id", "entry_date"])
    op.create_index("ix_journal_entries_shared", "journal_entries", ["shared_with_counselor"])

    op.create_table(
        "model_runtime_health",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_registry_id", sa.Integer(), nullable=True),
        sa.Column("modality", sa.String(), nullable=False),
        sa.Column("health_state", sa.String(), nullable=False),
        sa.Column("loader_status", sa.String(), nullable=True),
        sa.Column("preprocessing_status", sa.String(), nullable=True),
        sa.Column("smoke_test_status", sa.String(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("last_successful_inference_at", sa.DateTime(), nullable=True),
        sa.Column("last_failed_inference_at", sa.DateTime(), nullable=True),
        sa.Column("average_inference_duration_ms", sa.Float(), nullable=True),
        sa.Column("predictions_last_24h", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fusion_eligible", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["model_registry_id"], ["model_registry.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_runtime_health_id", "model_runtime_health", ["id"])
    op.create_index("ix_model_runtime_health_model_created", "model_runtime_health", ["model_registry_id", "created_at"])
    op.create_index("ix_model_runtime_health_modality_state", "model_runtime_health", ["modality", "health_state"])


def downgrade() -> None:
    op.drop_index("ix_model_runtime_health_modality_state", table_name="model_runtime_health")
    op.drop_index("ix_model_runtime_health_model_created", table_name="model_runtime_health")
    op.drop_index("ix_model_runtime_health_id", table_name="model_runtime_health")
    op.drop_table("model_runtime_health")

    op.drop_index("ix_journal_entries_shared", table_name="journal_entries")
    op.drop_index("ix_journal_entries_student_entry_date", table_name="journal_entries")
    op.drop_index("ix_journal_entries_student_created", table_name="journal_entries")
    op.drop_index("ix_journal_entries_journal_entry_id", table_name="journal_entries")
    op.drop_index("ix_journal_entries_id", table_name="journal_entries")
    op.drop_table("journal_entries")

    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_constraint("fk_chat_messages_ai_prediction", type_="foreignkey")
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("ai_prediction_id")
        batch_op.drop_column("ai_analysis_status")
        batch_op.drop_column("ai_analysis_requested")

    with op.batch_alter_table("safetalk_conversations") as batch_op:
        batch_op.drop_column("legacy_response_version")
        batch_op.drop_column("response_policy_version")

    with op.batch_alter_table("safetalk_bot_messages") as batch_op:
        batch_op.drop_column("response_variant_id")
        batch_op.drop_column("response_policy_version")

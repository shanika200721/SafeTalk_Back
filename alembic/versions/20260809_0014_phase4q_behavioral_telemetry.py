"""phase 4q behavioral telemetry aggregates

Revision ID: 20260809_0014
Revises: 20260809_0013
Create Date: 2026-08-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20260809_0014"
down_revision = "20260809_0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "behavioral_telemetry_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("source_page", sa.String(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("session_duration_seconds", sa.Float(), nullable=True),
        sa.Column("interaction_count", sa.Integer(), nullable=True),
        sa.Column("response_latency_ms", sa.Float(), nullable=True),
        sa.Column("typing_active_ms", sa.Float(), nullable=True),
        sa.Column("typing_pause_count", sa.Integer(), nullable=True),
        sa.Column("typed_character_count", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("consent_policy_version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('session_summary', 'interaction_summary', 'response_summary', 'typing_summary')",
            name="ck_behavioral_telemetry_event_type",
        ),
        sa.CheckConstraint(
            "session_duration_seconds IS NULL OR session_duration_seconds >= 0",
            name="ck_behavioral_telemetry_session_duration_nonnegative",
        ),
        sa.CheckConstraint(
            "interaction_count IS NULL OR interaction_count >= 0",
            name="ck_behavioral_telemetry_interaction_count_nonnegative",
        ),
        sa.CheckConstraint(
            "response_latency_ms IS NULL OR response_latency_ms >= 0",
            name="ck_behavioral_telemetry_response_latency_nonnegative",
        ),
        sa.CheckConstraint(
            "typing_active_ms IS NULL OR typing_active_ms >= 0",
            name="ck_behavioral_telemetry_typing_active_nonnegative",
        ),
        sa.CheckConstraint(
            "typing_pause_count IS NULL OR typing_pause_count >= 0",
            name="ck_behavioral_telemetry_typing_pause_nonnegative",
        ),
        sa.CheckConstraint(
            "typed_character_count IS NULL OR typed_character_count >= 0",
            name="ck_behavioral_telemetry_typed_chars_nonnegative",
        ),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_behavioral_telemetry_events_id", "behavioral_telemetry_events", ["id"])
    op.create_index(
        "ix_behavioral_telemetry_student_created",
        "behavioral_telemetry_events",
        ["student_id", "created_at"],
    )
    op.create_index(
        "ix_behavioral_telemetry_student_event",
        "behavioral_telemetry_events",
        ["student_id", "event_type", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_behavioral_telemetry_student_event", table_name="behavioral_telemetry_events")
    op.drop_index("ix_behavioral_telemetry_student_created", table_name="behavioral_telemetry_events")
    op.drop_index("ix_behavioral_telemetry_events_id", table_name="behavioral_telemetry_events")
    op.drop_table("behavioral_telemetry_events")

"""add versioned consent records

Revision ID: 20260725_0002
Revises: 20260714_0001
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0002"
down_revision = "20260714_0001"
branch_labels = None
depends_on = None


CONSENT_TYPE_CHECK = (
    "consent_type IN ('profile_processing', 'dass21_processing', 'mood_processing', "
    "'text_processing', 'voice_processing', 'face_processing', 'behavioral_processing', "
    "'counselor_escalation', 'research_data_use')"
)


def upgrade() -> None:
    op.create_table(
        "consent_records",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("consent_type", sa.String(), nullable=False),
        sa.Column("is_granted", sa.Boolean(), nullable=False),
        sa.Column("policy_version", sa.String(), nullable=False),
        sa.Column("granted_at", sa.DateTime(), nullable=True),
        sa.Column("withdrawn_at", sa.DateTime(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(CONSENT_TYPE_CHECK, name="ck_consent_records_known_type"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_consent_records_id"), "consent_records", ["id"], unique=False)
    op.create_index("ix_consent_records_type", "consent_records", ["consent_type"], unique=False)
    op.create_index(
        "ix_consent_records_user_type_created",
        "consent_records",
        ["user_id", "consent_type", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_consent_records_user_type_created", table_name="consent_records")
    op.drop_index("ix_consent_records_type", table_name="consent_records")
    op.drop_index(op.f("ix_consent_records_id"), table_name="consent_records")
    op.drop_table("consent_records")

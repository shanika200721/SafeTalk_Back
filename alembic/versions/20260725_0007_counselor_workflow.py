"""add counselor assignment review and note workflow

Revision ID: 20260725_0007
Revises: 20260725_0006
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0007"
down_revision = "20260725_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "counselor_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.String(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("counselor_id", sa.Integer(), nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=True),
        sa.Column("assignment_reason", sa.Text(), nullable=True),
        sa.Column("assigned_date", sa.DateTime(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("end_date", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["counselor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assignment_id"),
    )
    op.create_index("ix_counselor_assignments_id", "counselor_assignments", ["id"])
    op.create_index("ix_counselor_assignments_assignment_id", "counselor_assignments", ["assignment_id"])
    op.create_index("ix_counselor_assignments_student_active", "counselor_assignments", ["student_id", "active"])
    op.create_index("ix_counselor_assignments_counselor_active", "counselor_assignments", ["counselor_id", "active"])
    op.create_index("ix_counselor_assignments_assigned_date", "counselor_assignments", ["assigned_date"])

    op.create_table(
        "counselor_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("review_id", sa.String(), nullable=False),
        sa.Column("assessment_id", sa.Integer(), nullable=True),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("counselor_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="NEW"),
        sa.Column("review_notes", sa.Text(), nullable=True),
        sa.Column("decision", sa.Text(), nullable=True),
        sa.Column("risk_judgement", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "status IN ('NEW', 'UNDER_REVIEW', 'FOLLOW_UP_REQUIRED', 'REFERRED', 'CLOSED')",
            name="ck_counselor_reviews_status",
        ),
        sa.ForeignKeyConstraint(["assessment_id"], ["risk_assessments.id"]),
        sa.ForeignKeyConstraint(["counselor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("review_id"),
    )
    op.create_index("ix_counselor_reviews_id", "counselor_reviews", ["id"])
    op.create_index("ix_counselor_reviews_review_id", "counselor_reviews", ["review_id"])
    op.create_index("ix_counselor_reviews_student_status", "counselor_reviews", ["student_id", "status"])
    op.create_index("ix_counselor_reviews_counselor_status", "counselor_reviews", ["counselor_id", "status"])
    op.create_index("ix_counselor_reviews_assessment", "counselor_reviews", ["assessment_id"])

    op.create_table(
        "counselor_notes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("note_id", sa.String(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("counselor_id", sa.Integer(), nullable=False),
        sa.Column("note_text", sa.Text(), nullable=False),
        sa.Column("note_type", sa.String(), nullable=False, server_default="clinical"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["counselor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("note_id"),
    )
    op.create_index("ix_counselor_notes_id", "counselor_notes", ["id"])
    op.create_index("ix_counselor_notes_note_id", "counselor_notes", ["note_id"])
    op.create_index("ix_counselor_notes_student_active_created", "counselor_notes", ["student_id", "active", "created_at"])
    op.create_index("ix_counselor_notes_counselor_created", "counselor_notes", ["counselor_id", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_counselor_notes_counselor_created", table_name="counselor_notes")
    op.drop_index("ix_counselor_notes_student_active_created", table_name="counselor_notes")
    op.drop_index("ix_counselor_notes_note_id", table_name="counselor_notes")
    op.drop_index("ix_counselor_notes_id", table_name="counselor_notes")
    op.drop_table("counselor_notes")

    op.drop_index("ix_counselor_reviews_assessment", table_name="counselor_reviews")
    op.drop_index("ix_counselor_reviews_counselor_status", table_name="counselor_reviews")
    op.drop_index("ix_counselor_reviews_student_status", table_name="counselor_reviews")
    op.drop_index("ix_counselor_reviews_review_id", table_name="counselor_reviews")
    op.drop_index("ix_counselor_reviews_id", table_name="counselor_reviews")
    op.drop_table("counselor_reviews")

    op.drop_index("ix_counselor_assignments_assigned_date", table_name="counselor_assignments")
    op.drop_index("ix_counselor_assignments_counselor_active", table_name="counselor_assignments")
    op.drop_index("ix_counselor_assignments_student_active", table_name="counselor_assignments")
    op.drop_index("ix_counselor_assignments_assignment_id", table_name="counselor_assignments")
    op.drop_index("ix_counselor_assignments_id", table_name="counselor_assignments")
    op.drop_table("counselor_assignments")

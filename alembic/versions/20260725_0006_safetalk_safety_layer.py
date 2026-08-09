"""add SafeTalk deterministic safety audit fields

Revision ID: 20260725_0006
Revises: 20260725_0005
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0006"
down_revision = "20260725_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "safetalk_conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("topic_label", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("context_state", sa.JSON(), nullable=True),
        sa.Column("safety_policy_version", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("archived_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_safetalk_conversations_id", "safetalk_conversations", ["id"])
    op.create_index("ix_safetalk_conversations_status", "safetalk_conversations", ["status"])
    op.create_index(
        "ix_safetalk_conversations_user_created",
        "safetalk_conversations",
        ["user_id", "created_at"],
    )

    with op.batch_alter_table("safetalk_bot_messages") as batch_op:
        batch_op.add_column(sa.Column("conversation_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("route", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("severity", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("topic_label", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("response_template_version", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("safety_check_required", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("human_contact_recommended", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("safety_policy_version", sa.String(), nullable=True))
        batch_op.create_foreign_key(
            "fk_safetalk_bot_messages_conversation_id",
            "safetalk_conversations",
            ["conversation_id"],
            ["id"],
        )
        batch_op.create_index("ix_safetalk_bot_messages_conversation", ["conversation_id"])
        batch_op.create_index("ix_safetalk_bot_messages_route", ["route"])


def downgrade() -> None:
    # Export Phase 4F SafeTalk rows first if the safety audit fields are needed.
    with op.batch_alter_table("safetalk_bot_messages") as batch_op:
        batch_op.drop_index("ix_safetalk_bot_messages_route")
        batch_op.drop_index("ix_safetalk_bot_messages_conversation")
        batch_op.drop_constraint("fk_safetalk_bot_messages_conversation_id", type_="foreignkey")
        batch_op.drop_column("safety_policy_version")
        batch_op.drop_column("human_contact_recommended")
        batch_op.drop_column("safety_check_required")
        batch_op.drop_column("response_template_version")
        batch_op.drop_column("topic_label")
        batch_op.drop_column("severity")
        batch_op.drop_column("route")
        batch_op.drop_column("conversation_id")

    op.drop_index("ix_safetalk_conversations_user_created", table_name="safetalk_conversations")
    op.drop_index("ix_safetalk_conversations_status", table_name="safetalk_conversations")
    op.drop_index("ix_safetalk_conversations_id", table_name="safetalk_conversations")
    op.drop_table("safetalk_conversations")

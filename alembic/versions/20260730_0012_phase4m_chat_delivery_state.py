"""phase 4m chat delivery state

Revision ID: 20260730_0012
Revises: 20260725_0011
Create Date: 2026-07-30
"""

from alembic import op
import sqlalchemy as sa


revision = "20260730_0012"
down_revision = "20260725_0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.add_column(sa.Column("sent_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("delivered_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("delivery_status", sa.String(), nullable=False, server_default="sent"))

    op.execute("UPDATE chat_messages SET sent_at = COALESCE(sent_at, created_at)")
    op.execute("UPDATE chat_messages SET delivery_status = CASE WHEN is_read IS TRUE THEN 'read' ELSE 'sent' END")

    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.alter_column("delivery_status", server_default=None)


def downgrade() -> None:
    with op.batch_alter_table("chat_messages") as batch_op:
        batch_op.drop_column("delivery_status")
        batch_op.drop_column("delivered_at")
        batch_op.drop_column("sent_at")

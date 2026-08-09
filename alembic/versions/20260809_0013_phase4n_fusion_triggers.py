"""phase 4n fusion trigger idempotency

Revision ID: 20260809_0013
Revises: 20260730_0012
Create Date: 2026-08-09
"""

from alembic import op
import sqlalchemy as sa


revision = "20260809_0013"
down_revision = "20260730_0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("risk_assessments") as batch_op:
        batch_op.add_column(sa.Column("trigger_source", sa.String(), nullable=True))
        batch_op.add_column(sa.Column("trigger_prediction_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("trigger_metadata_json", sa.JSON(), nullable=True))
        batch_op.create_foreign_key(
            "fk_risk_assessments_trigger_prediction_id_modality_predictions",
            "modality_predictions",
            ["trigger_prediction_id"],
            ["id"],
            ondelete="SET NULL",
        )

    op.create_index(
        "ux_risk_assessments_trigger_prediction",
        "risk_assessments",
        ["trigger_prediction_id"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_risk_assessments_trigger_prediction", table_name="risk_assessments")
    with op.batch_alter_table("risk_assessments") as batch_op:
        batch_op.drop_constraint(
            "fk_risk_assessments_trigger_prediction_id_modality_predictions",
            type_="foreignkey",
        )
        batch_op.drop_column("trigger_metadata_json")
        batch_op.drop_column("trigger_prediction_id")
        batch_op.drop_column("trigger_source")

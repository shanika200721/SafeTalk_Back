"""add administration portal governance tables

Revision ID: 20260725_0009
Revises: 20260725_0008
Create Date: 2026-07-25
"""

from alembic import op
import sqlalchemy as sa


revision = "20260725_0009"
down_revision = "20260725_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("suspended_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("last_login_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("invitation_sent_at", sa.DateTime(), nullable=True))
        batch_op.create_index("ix_users_last_login_at", ["last_login_at"])

    with op.batch_alter_table("resources") as batch_op:
        batch_op.add_column(sa.Column("resource_type", sa.String(), nullable=False, server_default="article"))
        batch_op.add_column(sa.Column("status", sa.String(), nullable=False, server_default="draft"))
        batch_op.add_column(sa.Column("approved_by", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("approved_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("archived_at", sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column("metadata_json", sa.JSON(), nullable=True))
        batch_op.create_foreign_key("fk_resources_approved_by", "users", ["approved_by"], ["id"])
        batch_op.create_index("ix_resources_resource_type", ["resource_type"])
        batch_op.create_index("ix_resources_status", ["status"])

    op.create_table(
        "admin_audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("audit_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("entity_id", sa.String(), nullable=True),
        sa.Column("old_value_json", sa.JSON(), nullable=True),
        sa.Column("new_value_json", sa.JSON(), nullable=True),
        sa.Column("ip_address", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("privacy_scope", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("status IN ('success', 'failed', 'blocked')", name="ck_admin_audit_logs_status"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audit_id"),
    )
    op.create_index("ix_admin_audit_logs_id", "admin_audit_logs", ["id"])
    op.create_index("ix_admin_audit_logs_audit_id", "admin_audit_logs", ["audit_id"])
    op.create_index("ix_admin_audit_logs_created", "admin_audit_logs", ["created_at"])
    op.create_index("ix_admin_audit_logs_user_action", "admin_audit_logs", ["user_id", "action"])
    op.create_index("ix_admin_audit_logs_entity", "admin_audit_logs", ["entity_type", "entity_id"])

    op.create_table(
        "system_settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("section", sa.String(), nullable=False),
        sa.Column("setting_key", sa.String(), nullable=False),
        sa.Column("setting_value", sa.JSON(), nullable=True),
        sa.Column("value_type", sa.String(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("read_only", sa.Boolean(), nullable=False),
        sa.Column("updated_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["updated_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("section", "setting_key", name="uq_system_settings_section_key"),
    )
    op.create_index("ix_system_settings_id", "system_settings", ["id"])
    op.create_index("ix_system_settings_section", "system_settings", ["section"])

    op.create_table(
        "admin_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("report_id", sa.String(), nullable=False),
        sa.Column("report_type", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("parameters_json", sa.JSON(), nullable=True),
        sa.Column("summary_json", sa.JSON(), nullable=True),
        sa.Column("export_formats_json", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("requested_by", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["requested_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("report_id"),
    )
    op.create_index("ix_admin_reports_id", "admin_reports", ["id"])
    op.create_index("ix_admin_reports_report_id", "admin_reports", ["report_id"])
    op.create_index("ix_admin_reports_type_created", "admin_reports", ["report_type", "created_at"])
    op.create_index("ix_admin_reports_requested_by", "admin_reports", ["requested_by"])


def downgrade() -> None:
    op.drop_index("ix_admin_reports_requested_by", table_name="admin_reports")
    op.drop_index("ix_admin_reports_type_created", table_name="admin_reports")
    op.drop_index("ix_admin_reports_report_id", table_name="admin_reports")
    op.drop_index("ix_admin_reports_id", table_name="admin_reports")
    op.drop_table("admin_reports")

    op.drop_index("ix_system_settings_section", table_name="system_settings")
    op.drop_index("ix_system_settings_id", table_name="system_settings")
    op.drop_table("system_settings")

    op.drop_index("ix_admin_audit_logs_entity", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_user_action", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_created", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_audit_id", table_name="admin_audit_logs")
    op.drop_index("ix_admin_audit_logs_id", table_name="admin_audit_logs")
    op.drop_table("admin_audit_logs")

    with op.batch_alter_table("resources") as batch_op:
        batch_op.drop_index("ix_resources_status")
        batch_op.drop_index("ix_resources_resource_type")
        batch_op.drop_constraint("fk_resources_approved_by", type_="foreignkey")
        batch_op.drop_column("metadata_json")
        batch_op.drop_column("archived_at")
        batch_op.drop_column("approved_at")
        batch_op.drop_column("approved_by")
        batch_op.drop_column("status")
        batch_op.drop_column("resource_type")

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_index("ix_users_last_login_at")
        batch_op.drop_column("invitation_sent_at")
        batch_op.drop_column("last_login_at")
        batch_op.drop_column("suspended_at")

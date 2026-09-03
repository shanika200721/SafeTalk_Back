"""add university counselor directory and support contacts

Revision ID: 20260725_0008
Revises: 20260725_0007
Create Date: 2026-07-25
"""

from datetime import datetime

from alembic import op
import sqlalchemy as sa


revision = "20260725_0008"
down_revision = "20260725_0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "universities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("university_id", sa.String(), nullable=False),
        sa.Column("university_name", sa.String(), nullable=False),
        sa.Column("university_code", sa.String(), nullable=False),
        sa.Column("campus_name", sa.String(), nullable=True),
        sa.Column("address", sa.Text(), nullable=True),
        sa.Column("district", sa.String(), nullable=True),
        sa.Column("province", sa.String(), nullable=True),
        sa.Column("general_phone", sa.String(), nullable=True),
        sa.Column("counseling_unit_phone", sa.String(), nullable=True),
        sa.Column("emergency_support_phone", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("website", sa.String(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("university_code", "campus_name", name="uq_universities_code_campus"),
        sa.UniqueConstraint("university_id"),
    )
    op.create_index("ix_universities_id", "universities", ["id"])
    op.create_index("ix_universities_university_id", "universities", ["university_id"])
    op.create_index("ix_universities_active", "universities", ["active"])
    op.create_index("ix_universities_code", "universities", ["university_code"])

    with op.batch_alter_table("users") as batch_op:
        batch_op.add_column(sa.Column("university_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key("fk_users_university_id", "universities", ["university_id"], ["id"])

    op.create_table(
        "counselor_profiles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("counselor_profile_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("university_id", sa.Integer(), nullable=True),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("professional_title", sa.String(), nullable=True),
        sa.Column("qualification", sa.Text(), nullable=True),
        sa.Column("specialization", sa.Text(), nullable=True),
        sa.Column("registration_number", sa.String(), nullable=True),
        sa.Column("office_name", sa.String(), nullable=True),
        sa.Column("office_location", sa.Text(), nullable=True),
        sa.Column("telephone_number", sa.String(), nullable=True),
        sa.Column("whatsapp_number", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("available_days", sa.String(), nullable=True),
        sa.Column("available_from", sa.String(), nullable=True),
        sa.Column("available_until", sa.String(), nullable=True),
        sa.Column("accepts_voice_calls", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("accepts_whatsapp_calls", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("accepts_whatsapp_messages", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("emergency_contact_enabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("profile_photo_reference", sa.String(), nullable=True),
        sa.Column("languages_json", sa.JSON(), nullable=True),
        sa.Column("availability_status", sa.String(), nullable=False, server_default="available"),
        sa.Column("approved", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("student_visible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("admin_notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["university_id"], ["universities.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("counselor_profile_id"),
        sa.UniqueConstraint("user_id", name="uq_counselor_profiles_user_id"),
    )
    op.create_index("ix_counselor_profiles_id", "counselor_profiles", ["id"])
    op.create_index("ix_counselor_profiles_counselor_profile_id", "counselor_profiles", ["counselor_profile_id"])
    op.create_index("ix_counselor_profiles_university_active", "counselor_profiles", ["university_id", "active"])
    op.create_index("ix_counselor_profiles_availability", "counselor_profiles", ["availability_status"])

    op.create_table(
        "counselor_university_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.String(), nullable=False),
        sa.Column("counselor_profile_id", sa.Integer(), nullable=False),
        sa.Column("university_id", sa.Integer(), nullable=False),
        sa.Column("assigned_by", sa.Integer(), nullable=True),
        sa.Column("assignment_reason", sa.Text(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("assigned_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["assigned_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["counselor_profile_id"], ["counselor_profiles.id"]),
        sa.ForeignKeyConstraint(["university_id"], ["universities.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("assignment_id"),
    )
    op.create_index("ix_counselor_university_assignments_id", "counselor_university_assignments", ["id"])
    op.create_index("ix_counselor_university_assignments_assignment_id", "counselor_university_assignments", ["assignment_id"])
    op.create_index(
        "ix_counselor_university_assignments_profile_active",
        "counselor_university_assignments",
        ["counselor_profile_id", "active"],
    )
    op.create_index(
        "ix_counselor_university_assignments_university_active",
        "counselor_university_assignments",
        ["university_id", "active"],
    )

    op.create_table(
        "support_contacts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("support_contact_id", sa.String(), nullable=False),
        sa.Column("university_id", sa.Integer(), nullable=True),
        sa.Column("counselor_profile_id", sa.Integer(), nullable=True),
        sa.Column("contact_type", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column("telephone_number", sa.String(), nullable=True),
        sa.Column("whatsapp_number", sa.String(), nullable=True),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("available_days", sa.String(), nullable=True),
        sa.Column("available_from", sa.String(), nullable=True),
        sa.Column("available_until", sa.String(), nullable=True),
        sa.Column("telephone_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("whatsapp_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("student_visible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("emergency_service", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("verified_at", sa.DateTime(), nullable=True),
        sa.Column("verified_by", sa.Integer(), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "contact_type IN ('assigned_counselor', 'university_unit', 'university_fallback', 'system_fallback', 'crisis_service')",
            name="ck_support_contacts_type",
        ),
        sa.ForeignKeyConstraint(["counselor_profile_id"], ["counselor_profiles.id"]),
        sa.ForeignKeyConstraint(["university_id"], ["universities.id"]),
        sa.ForeignKeyConstraint(["verified_by"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("support_contact_id"),
    )
    op.create_index("ix_support_contacts_id", "support_contacts", ["id"])
    op.create_index("ix_support_contacts_support_contact_id", "support_contacts", ["support_contact_id"])
    op.create_index("ix_support_contacts_university_active_priority", "support_contacts", ["university_id", "active", "priority"])
    op.create_index("ix_support_contacts_type_active", "support_contacts", ["contact_type", "active"])

    op.create_table(
        "support_contact_actions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("action_id", sa.String(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("support_contact_id", sa.Integer(), nullable=True),
        sa.Column("contact_type", sa.String(), nullable=True),
        sa.Column("action_type", sa.String(), nullable=False),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "action_type IN ('support_panel_opened', 'telephone_action_selected', 'whatsapp_action_selected', 'number_copied', 'support_details_viewed')",
            name="ck_support_contact_actions_type",
        ),
        sa.ForeignKeyConstraint(["support_contact_id"], ["support_contacts.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("action_id"),
    )
    op.create_index("ix_support_contact_actions_id", "support_contact_actions", ["id"])
    op.create_index("ix_support_contact_actions_action_id", "support_contact_actions", ["action_id"])
    op.create_index("ix_support_contact_actions_user_created", "support_contact_actions", ["user_id", "created_at"])
    op.create_index("ix_support_contact_actions_contact_created", "support_contact_actions", ["support_contact_id", "created_at"])

    op.create_table(
        "counselor_profile_audit",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("audit_id", sa.String(), nullable=False),
        sa.Column("counselor_profile_id", sa.Integer(), nullable=True),
        sa.Column("changed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("change_type", sa.String(), nullable=False),
        sa.Column("changed_fields", sa.JSON(), nullable=True),
        sa.Column("before_json", sa.JSON(), nullable=True),
        sa.Column("after_json", sa.JSON(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["changed_by_user_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["counselor_profile_id"], ["counselor_profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("audit_id"),
    )
    op.create_index("ix_counselor_profile_audit_id", "counselor_profile_audit", ["id"])
    op.create_index("ix_counselor_profile_audit_audit_id", "counselor_profile_audit", ["audit_id"])
    op.create_index("ix_counselor_profile_audit_profile_created", "counselor_profile_audit", ["counselor_profile_id", "created_at"])
    op.create_index("ix_counselor_profile_audit_changed_by", "counselor_profile_audit", ["changed_by_user_id"])

    now = datetime.utcnow()
    support_contacts = sa.table(
        "support_contacts",
        sa.column("support_contact_id", sa.String),
        sa.column("contact_type", sa.String),
        sa.column("display_name", sa.String),
        sa.column("telephone_number", sa.String),
        sa.column("whatsapp_number", sa.String),
        sa.column("telephone_enabled", sa.Boolean),
        sa.column("whatsapp_enabled", sa.Boolean),
        sa.column("student_visible", sa.Boolean),
        sa.column("emergency_service", sa.Boolean),
        sa.column("verified", sa.Boolean),
        sa.column("verified_at", sa.DateTime),
        sa.column("active", sa.Boolean),
        sa.column("priority", sa.Integer),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )
    op.bulk_insert(
        support_contacts,
        [
            {
                "support_contact_id": "support-fallback-safetalk-v1",
                "contact_type": "system_fallback",
                "display_name": "SafeTalk Counselor Support",
                "telephone_number": "+94705584634",
                "whatsapp_number": "+94705584634",
                "telephone_enabled": True,
                "whatsapp_enabled": True,
                "student_visible": True,
                "emergency_service": False,
                "verified": True,
                "verified_at": now,
                "active": True,
                "priority": 1000,
                "created_at": now,
                "updated_at": now,
            }
        ],
    )


def downgrade() -> None:
    op.drop_index("ix_counselor_profile_audit_changed_by", table_name="counselor_profile_audit")
    op.drop_index("ix_counselor_profile_audit_profile_created", table_name="counselor_profile_audit")
    op.drop_index("ix_counselor_profile_audit_audit_id", table_name="counselor_profile_audit")
    op.drop_index("ix_counselor_profile_audit_id", table_name="counselor_profile_audit")
    op.drop_table("counselor_profile_audit")

    op.drop_index("ix_support_contact_actions_contact_created", table_name="support_contact_actions")
    op.drop_index("ix_support_contact_actions_user_created", table_name="support_contact_actions")
    op.drop_index("ix_support_contact_actions_action_id", table_name="support_contact_actions")
    op.drop_index("ix_support_contact_actions_id", table_name="support_contact_actions")
    op.drop_table("support_contact_actions")

    op.drop_index("ix_support_contacts_type_active", table_name="support_contacts")
    op.drop_index("ix_support_contacts_university_active_priority", table_name="support_contacts")
    op.drop_index("ix_support_contacts_support_contact_id", table_name="support_contacts")
    op.drop_index("ix_support_contacts_id", table_name="support_contacts")
    op.drop_table("support_contacts")

    op.drop_index("ix_counselor_university_assignments_university_active", table_name="counselor_university_assignments")
    op.drop_index("ix_counselor_university_assignments_profile_active", table_name="counselor_university_assignments")
    op.drop_index("ix_counselor_university_assignments_assignment_id", table_name="counselor_university_assignments")
    op.drop_index("ix_counselor_university_assignments_id", table_name="counselor_university_assignments")
    op.drop_table("counselor_university_assignments")

    op.drop_index("ix_counselor_profiles_availability", table_name="counselor_profiles")
    op.drop_index("ix_counselor_profiles_university_active", table_name="counselor_profiles")
    op.drop_index("ix_counselor_profiles_counselor_profile_id", table_name="counselor_profiles")
    op.drop_index("ix_counselor_profiles_id", table_name="counselor_profiles")
    op.drop_table("counselor_profiles")

    with op.batch_alter_table("users") as batch_op:
        batch_op.drop_constraint("fk_users_university_id", type_="foreignkey")
        batch_op.drop_column("university_id")

    op.drop_index("ix_universities_code", table_name="universities")
    op.drop_index("ix_universities_active", table_name="universities")
    op.drop_index("ix_universities_university_id", table_name="universities")
    op.drop_index("ix_universities_id", table_name="universities")
    op.drop_table("universities")

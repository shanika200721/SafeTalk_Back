"""initial schema with phase 1 ml tracking

Revision ID: 20260714_0001
Revises:
Create Date: 2026-07-14
"""

from alembic import op
import sqlalchemy as sa


revision = "20260714_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(), nullable=True),
        sa.Column("username", sa.String(), nullable=True),
        sa.Column("full_name", sa.String(), nullable=True),
        sa.Column("hashed_password", sa.String(), nullable=True),
        sa.Column("role", sa.Enum("STUDENT", "COUNSELOR", "ADMIN", "PSYCHIATRIST", name="userrole"), nullable=True),
        sa.Column("department", sa.String(), nullable=True),
        sa.Column("year_of_study", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)
    op.create_index(op.f("ix_users_id"), "users", ["id"], unique=False)
    op.create_index(op.f("ix_users_username"), "users", ["username"], unique=True)

    op.create_table(
        "model_registry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("modality", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("framework", sa.String(), nullable=False),
        sa.Column("artifact_path", sa.String(), nullable=False),
        sa.Column("preprocessing_path", sa.String(), nullable=True),
        sa.Column("dataset_version", sa.String(), nullable=True),
        sa.Column("feature_schema_version", sa.String(), nullable=True),
        sa.Column("metrics_json", sa.JSON(), nullable=True),
        sa.Column("thresholds_json", sa.JSON(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_name", "modality", "version", name="uq_model_registry_name_modality_version"),
    )
    op.create_index(op.f("ix_model_registry_created_at"), "model_registry", ["created_at"], unique=False)
    op.create_index(op.f("ix_model_registry_id"), "model_registry", ["id"], unique=False)
    op.create_index(op.f("ix_model_registry_is_active"), "model_registry", ["is_active"], unique=False)
    op.create_index("ix_model_registry_modality", "model_registry", ["modality"], unique=False)
    op.create_index("ix_model_registry_model_version", "model_registry", ["model_name", "version"], unique=False)
    op.create_index(
        "uq_model_registry_one_active",
        "model_registry",
        ["model_name", "modality"],
        unique=True,
        postgresql_where=sa.text("is_active = true"),
        sqlite_where=sa.text("is_active = 1"),
    )

    op.create_table(
        "assessment_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("assessment_type", sa.String(), nullable=True),
        sa.Column("data", sa.JSON(), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_assessment_history_id"), "assessment_history", ["id"], unique=False)

    op.create_table(
        "assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("assessment_type", sa.String(), nullable=True),
        sa.Column("profile_score", sa.Float(), nullable=True),
        sa.Column("mood_score", sa.Float(), nullable=True),
        sa.Column("dass21_score", sa.Float(), nullable=True),
        sa.Column("text_score", sa.Float(), nullable=True),
        sa.Column("voice_score", sa.Float(), nullable=True),
        sa.Column("face_score", sa.Float(), nullable=True),
        sa.Column("behavioral_score", sa.Float(), nullable=True),
        sa.Column("composite_score", sa.Float(), nullable=True),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("needs_escalation", sa.Boolean(), nullable=True),
        sa.Column("recommendations", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_assessments_id"), "assessments", ["id"], unique=False)

    op.create_table(
        "daily_checkins",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("mood", sa.Integer(), nullable=True),
        sa.Column("mood_description", sa.String(), nullable=True),
        sa.Column("sleep_hours", sa.Float(), nullable=True),
        sa.Column("exercise_minutes", sa.Integer(), nullable=True),
        sa.Column("social_interaction", sa.String(), nullable=True),
        sa.Column("stress_level", sa.Integer(), nullable=True),
        sa.Column("anxiety_level", sa.Integer(), nullable=True),
        sa.Column("negative_thoughts", sa.Boolean(), nullable=True),
        sa.Column("substance_use_today", sa.Boolean(), nullable=True),
        sa.Column("self_harm_thoughts", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_daily_checkins_id"), "daily_checkins", ["id"], unique=False)

    op.create_table(
        "dass21_assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("responses", sa.JSON(), nullable=True),
        sa.Column("depression_score", sa.Float(), nullable=True),
        sa.Column("anxiety_score", sa.Float(), nullable=True),
        sa.Column("stress_score", sa.Float(), nullable=True),
        sa.Column("total_dass21_score", sa.Float(), nullable=True),
        sa.Column("depression_severity", sa.String(), nullable=True),
        sa.Column("anxiety_severity", sa.String(), nullable=True),
        sa.Column("stress_severity", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_dass21_assessments_id"), "dass21_assessments", ["id"], unique=False)

    op.create_table(
        "feature_snapshots",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("modality", sa.String(), nullable=False),
        sa.Column("features_json", sa.JSON(), nullable=False),
        sa.Column("preprocessing_version", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_feature_snapshots_created_at"), "feature_snapshots", ["created_at"], unique=False)
    op.create_index(op.f("ix_feature_snapshots_id"), "feature_snapshots", ["id"], unique=False)
    op.create_index("ix_feature_snapshots_modality", "feature_snapshots", ["modality"], unique=False)
    op.create_index("ix_feature_snapshots_source", "feature_snapshots", ["source_type", "source_record_id"], unique=False)
    op.create_index("ix_feature_snapshots_student_created", "feature_snapshots", ["student_id", "created_at"], unique=False)

    op.create_table(
        "profile_assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("gpa", sa.Float(), nullable=True),
        sa.Column("repeated_subjects", sa.Integer(), nullable=True),
        sa.Column("attendance", sa.Float(), nullable=True),
        sa.Column("academic_difficulty", sa.String(), nullable=True),
        sa.Column("family_relationship_score", sa.Float(), nullable=True),
        sa.Column("income_level", sa.String(), nullable=True),
        sa.Column("parents_employment", sa.String(), nullable=True),
        sa.Column("family_support", sa.Float(), nullable=True),
        sa.Column("living_arrangement", sa.String(), nullable=True),
        sa.Column("employment_status", sa.String(), nullable=True),
        sa.Column("financial_stress", sa.Boolean(), nullable=True),
        sa.Column("communication_skills", sa.Float(), nullable=True),
        sa.Column("social_connection", sa.Float(), nullable=True),
        sa.Column("sleep_pattern", sa.String(), nullable=True),
        sa.Column("exercise_frequency", sa.String(), nullable=True),
        sa.Column("substance_use", sa.String(), nullable=True),
        sa.Column("profile_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_profile_assessments_id"), "profile_assessments", ["id"], unique=False)

    op.create_table(
        "resources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("category", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("url", sa.String(), nullable=True),
        sa.Column("phone", sa.String(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_resources_id"), "resources", ["id"], unique=False)

    op.create_table(
        "alerts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("alert_type", sa.String(), nullable=True),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"], unique=False)
    op.create_index(op.f("ix_alerts_id"), "alerts", ["id"], unique=False)
    op.create_index("ix_alerts_is_read", "alerts", ["is_read"], unique=False)
    op.create_index("ix_alerts_risk_level", "alerts", ["risk_level"], unique=False)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sender_id", sa.Integer(), nullable=True),
        sa.Column("receiver_id", sa.Integer(), nullable=True),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("message_type", sa.String(), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=True),
        sa.Column("read_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["receiver_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["sender_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_messages_id"), "chat_messages", ["id"], unique=False)

    op.create_table(
        "counselor_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("counselor_id", sa.Integer(), nullable=True),
        sa.Column("session_type", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("risk_level_at_escalation", sa.String(), nullable=True),
        sa.Column("counselor_notes", sa.Text(), nullable=True),
        sa.Column("intervention_type", sa.String(), nullable=True),
        sa.Column("outcome", sa.String(), nullable=True),
        sa.Column("follow_up_needed", sa.Boolean(), nullable=True),
        sa.Column("follow_up_date", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["counselor_id"], ["users.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_counselor_sessions_id"), "counselor_sessions", ["id"], unique=False)

    op.create_table(
        "modality_predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("modality", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("model_registry_id", sa.Integer(), nullable=False),
        sa.Column("predicted_class", sa.String(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("score_0_100", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("explanation_json", sa.JSON(), nullable=True),
        sa.Column("processing_time_ms", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_modality_predictions_confidence_0_1"),
        sa.CheckConstraint("probability >= 0 AND probability <= 1", name="ck_modality_predictions_probability_0_1"),
        sa.CheckConstraint("score_0_100 >= 0 AND score_0_100 <= 100", name="ck_modality_predictions_score_0_100"),
        sa.ForeignKeyConstraint(["model_registry_id"], ["model_registry.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_modality_predictions_created_at"), "modality_predictions", ["created_at"], unique=False)
    op.create_index(op.f("ix_modality_predictions_id"), "modality_predictions", ["id"], unique=False)
    op.create_index("ix_modality_predictions_modality", "modality_predictions", ["modality"], unique=False)
    op.create_index("ix_modality_predictions_source", "modality_predictions", ["source_type", "source_record_id"], unique=False)
    op.create_index("ix_modality_predictions_student_created", "modality_predictions", ["student_id", "created_at"], unique=False)

    op.create_table(
        "risk_assessments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("fusion_model_id", sa.Integer(), nullable=True),
        sa.Column("final_probability", sa.Float(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("data_completeness", sa.JSON(), nullable=True),
        sa.Column("safety_override", sa.Boolean(), nullable=False),
        sa.Column("safety_override_reason", sa.Text(), nullable=True),
        sa.Column("explanation_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name="ck_risk_assessments_confidence_0_1"),
        sa.CheckConstraint("final_probability >= 0 AND final_probability <= 1", name="ck_risk_assessments_final_probability_0_1"),
        sa.CheckConstraint("final_score >= 0 AND final_score <= 100", name="ck_risk_assessments_final_score_0_100"),
        sa.ForeignKeyConstraint(["fusion_model_id"], ["model_registry.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["student_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_risk_assessments_created_at"), "risk_assessments", ["created_at"], unique=False)
    op.create_index(op.f("ix_risk_assessments_id"), "risk_assessments", ["id"], unique=False)
    op.create_index("ix_risk_assessments_risk_level", "risk_assessments", ["risk_level"], unique=False)
    op.create_index("ix_risk_assessments_safety_override", "risk_assessments", ["safety_override"], unique=False)
    op.create_index("ix_risk_assessments_student_created", "risk_assessments", ["student_id", "created_at"], unique=False)

    op.create_table(
        "safetalk_bot_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_message", sa.Text(), nullable=True),
        sa.Column("bot_response", sa.Text(), nullable=True),
        sa.Column("intent", sa.String(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("crisis_level", sa.Integer(), nullable=True),
        sa.Column("response_details", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_safetalk_bot_messages_id"), "safetalk_bot_messages", ["id"], unique=False)

    op.create_table(
        "worker_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("job_type", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("source_record_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_worker_jobs_created_at", "worker_jobs", ["created_at"], unique=False)
    op.create_index(op.f("ix_worker_jobs_id"), "worker_jobs", ["id"], unique=False)
    op.create_index("ix_worker_jobs_source", "worker_jobs", ["source_type", "source_record_id"], unique=False)
    op.create_index("ix_worker_jobs_status", "worker_jobs", ["status"], unique=False)

    op.create_table(
        "alert_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("alert_id", sa.Integer(), nullable=False),
        sa.Column("old_status", sa.String(), nullable=True),
        sa.Column("new_status", sa.String(), nullable=False),
        sa.Column("changed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["alert_id"], ["alerts.id"]),
        sa.ForeignKeyConstraint(["changed_by_user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_alert_events_alert_created", "alert_events", ["alert_id", "created_at"], unique=False)
    op.create_index(op.f("ix_alert_events_created_at"), "alert_events", ["created_at"], unique=False)
    op.create_index(op.f("ix_alert_events_id"), "alert_events", ["id"], unique=False)
    op.create_index("ix_alert_events_new_status", "alert_events", ["new_status"], unique=False)

    op.create_table(
        "risk_assessment_inputs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("risk_assessment_id", sa.Integer(), nullable=False),
        sa.Column("modality_prediction_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(["modality_prediction_id"], ["modality_predictions.id"]),
        sa.ForeignKeyConstraint(["risk_assessment_id"], ["risk_assessments.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("risk_assessment_id", "modality_prediction_id", name="uq_risk_assessment_input_pair"),
    )
    op.create_index(op.f("ix_risk_assessment_inputs_id"), "risk_assessment_inputs", ["id"], unique=False)
    op.create_index("ix_risk_assessment_inputs_assessment", "risk_assessment_inputs", ["risk_assessment_id"], unique=False)
    op.create_index("ix_risk_assessment_inputs_prediction", "risk_assessment_inputs", ["modality_prediction_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_risk_assessment_inputs_prediction", table_name="risk_assessment_inputs")
    op.drop_index("ix_risk_assessment_inputs_assessment", table_name="risk_assessment_inputs")
    op.drop_index(op.f("ix_risk_assessment_inputs_id"), table_name="risk_assessment_inputs")
    op.drop_table("risk_assessment_inputs")
    op.drop_index("ix_alert_events_new_status", table_name="alert_events")
    op.drop_index(op.f("ix_alert_events_id"), table_name="alert_events")
    op.drop_index(op.f("ix_alert_events_created_at"), table_name="alert_events")
    op.drop_index("ix_alert_events_alert_created", table_name="alert_events")
    op.drop_table("alert_events")
    op.drop_index("ix_worker_jobs_status", table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_source", table_name="worker_jobs")
    op.drop_index(op.f("ix_worker_jobs_id"), table_name="worker_jobs")
    op.drop_index("ix_worker_jobs_created_at", table_name="worker_jobs")
    op.drop_table("worker_jobs")
    op.drop_index(op.f("ix_safetalk_bot_messages_id"), table_name="safetalk_bot_messages")
    op.drop_table("safetalk_bot_messages")
    op.drop_index("ix_risk_assessments_student_created", table_name="risk_assessments")
    op.drop_index("ix_risk_assessments_safety_override", table_name="risk_assessments")
    op.drop_index("ix_risk_assessments_risk_level", table_name="risk_assessments")
    op.drop_index(op.f("ix_risk_assessments_id"), table_name="risk_assessments")
    op.drop_index(op.f("ix_risk_assessments_created_at"), table_name="risk_assessments")
    op.drop_table("risk_assessments")
    op.drop_index("ix_modality_predictions_student_created", table_name="modality_predictions")
    op.drop_index("ix_modality_predictions_source", table_name="modality_predictions")
    op.drop_index("ix_modality_predictions_modality", table_name="modality_predictions")
    op.drop_index(op.f("ix_modality_predictions_id"), table_name="modality_predictions")
    op.drop_index(op.f("ix_modality_predictions_created_at"), table_name="modality_predictions")
    op.drop_table("modality_predictions")
    op.drop_index(op.f("ix_counselor_sessions_id"), table_name="counselor_sessions")
    op.drop_table("counselor_sessions")
    op.drop_index(op.f("ix_chat_messages_id"), table_name="chat_messages")
    op.drop_table("chat_messages")
    op.drop_index("ix_alerts_risk_level", table_name="alerts")
    op.drop_index("ix_alerts_is_read", table_name="alerts")
    op.drop_index(op.f("ix_alerts_id"), table_name="alerts")
    op.drop_index("ix_alerts_created_at", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index(op.f("ix_resources_id"), table_name="resources")
    op.drop_table("resources")
    op.drop_index(op.f("ix_profile_assessments_id"), table_name="profile_assessments")
    op.drop_table("profile_assessments")
    op.drop_index("ix_feature_snapshots_student_created", table_name="feature_snapshots")
    op.drop_index("ix_feature_snapshots_source", table_name="feature_snapshots")
    op.drop_index("ix_feature_snapshots_modality", table_name="feature_snapshots")
    op.drop_index(op.f("ix_feature_snapshots_id"), table_name="feature_snapshots")
    op.drop_index(op.f("ix_feature_snapshots_created_at"), table_name="feature_snapshots")
    op.drop_table("feature_snapshots")
    op.drop_index(op.f("ix_dass21_assessments_id"), table_name="dass21_assessments")
    op.drop_table("dass21_assessments")
    op.drop_index(op.f("ix_daily_checkins_id"), table_name="daily_checkins")
    op.drop_table("daily_checkins")
    op.drop_index(op.f("ix_assessments_id"), table_name="assessments")
    op.drop_table("assessments")
    op.drop_index(op.f("ix_assessment_history_id"), table_name="assessment_history")
    op.drop_table("assessment_history")
    op.drop_index("uq_model_registry_one_active", table_name="model_registry")
    op.drop_index("ix_model_registry_model_version", table_name="model_registry")
    op.drop_index("ix_model_registry_modality", table_name="model_registry")
    op.drop_index(op.f("ix_model_registry_is_active"), table_name="model_registry")
    op.drop_index(op.f("ix_model_registry_id"), table_name="model_registry")
    op.drop_index(op.f("ix_model_registry_created_at"), table_name="model_registry")
    op.drop_table("model_registry")
    op.drop_index(op.f("ix_users_username"), table_name="users")
    op.drop_index(op.f("ix_users_id"), table_name="users")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
    sa.Enum("STUDENT", "COUNSELOR", "ADMIN", "PSYCHIATRIST", name="userrole").drop(op.get_bind(), checkfirst=True)

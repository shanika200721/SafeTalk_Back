BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 20260714_0001

CREATE TYPE userrole AS ENUM ('STUDENT', 'COUNSELOR', 'ADMIN', 'PSYCHIATRIST');

CREATE TABLE users (
    id SERIAL NOT NULL, 
    email VARCHAR, 
    username VARCHAR, 
    full_name VARCHAR, 
    hashed_password VARCHAR, 
    role userrole, 
    department VARCHAR, 
    year_of_study INTEGER, 
    is_active BOOLEAN, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    updated_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id)
);

CREATE UNIQUE INDEX ix_users_email ON users (email);

CREATE INDEX ix_users_id ON users (id);

CREATE UNIQUE INDEX ix_users_username ON users (username);

CREATE TABLE model_registry (
    id SERIAL NOT NULL, 
    model_name VARCHAR NOT NULL, 
    modality VARCHAR NOT NULL, 
    version VARCHAR NOT NULL, 
    framework VARCHAR NOT NULL, 
    artifact_path VARCHAR NOT NULL, 
    preprocessing_path VARCHAR, 
    dataset_version VARCHAR, 
    feature_schema_version VARCHAR, 
    metrics_json JSON, 
    thresholds_json JSON, 
    is_active BOOLEAN NOT NULL, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_model_registry_name_modality_version UNIQUE (model_name, modality, version)
);

CREATE INDEX ix_model_registry_created_at ON model_registry (created_at);

CREATE INDEX ix_model_registry_id ON model_registry (id);

CREATE INDEX ix_model_registry_is_active ON model_registry (is_active);

CREATE INDEX ix_model_registry_modality ON model_registry (modality);

CREATE INDEX ix_model_registry_model_version ON model_registry (model_name, version);

CREATE UNIQUE INDEX uq_model_registry_one_active ON model_registry (model_name, modality) WHERE is_active = true;

CREATE TABLE assessment_history (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    assessment_type VARCHAR, 
    data JSON, 
    composite_score FLOAT, 
    risk_level VARCHAR, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_assessment_history_id ON assessment_history (id);

CREATE TABLE assessments (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    assessment_type VARCHAR, 
    profile_score FLOAT, 
    mood_score FLOAT, 
    dass21_score FLOAT, 
    text_score FLOAT, 
    voice_score FLOAT, 
    face_score FLOAT, 
    behavioral_score FLOAT, 
    composite_score FLOAT, 
    risk_level VARCHAR, 
    needs_escalation BOOLEAN, 
    recommendations JSON, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_assessments_id ON assessments (id);

CREATE TABLE daily_checkins (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    mood INTEGER, 
    mood_description VARCHAR, 
    sleep_hours FLOAT, 
    exercise_minutes INTEGER, 
    social_interaction VARCHAR, 
    stress_level INTEGER, 
    anxiety_level INTEGER, 
    negative_thoughts BOOLEAN, 
    substance_use_today BOOLEAN, 
    self_harm_thoughts BOOLEAN, 
    notes TEXT, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_daily_checkins_id ON daily_checkins (id);

CREATE TABLE dass21_assessments (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    responses JSON, 
    depression_score FLOAT, 
    anxiety_score FLOAT, 
    stress_score FLOAT, 
    total_dass21_score FLOAT, 
    depression_severity VARCHAR, 
    anxiety_severity VARCHAR, 
    stress_severity VARCHAR, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_dass21_assessments_id ON dass21_assessments (id);

CREATE TABLE feature_snapshots (
    id SERIAL NOT NULL, 
    student_id INTEGER NOT NULL, 
    source_type VARCHAR NOT NULL, 
    source_record_id INTEGER NOT NULL, 
    modality VARCHAR NOT NULL, 
    features_json JSON NOT NULL, 
    preprocessing_version VARCHAR, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(student_id) REFERENCES users (id)
);

CREATE INDEX ix_feature_snapshots_created_at ON feature_snapshots (created_at);

CREATE INDEX ix_feature_snapshots_id ON feature_snapshots (id);

CREATE INDEX ix_feature_snapshots_modality ON feature_snapshots (modality);

CREATE INDEX ix_feature_snapshots_source ON feature_snapshots (source_type, source_record_id);

CREATE INDEX ix_feature_snapshots_student_created ON feature_snapshots (student_id, created_at);

CREATE TABLE profile_assessments (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    gpa FLOAT, 
    repeated_subjects INTEGER, 
    attendance FLOAT, 
    academic_difficulty VARCHAR, 
    family_relationship_score FLOAT, 
    income_level VARCHAR, 
    parents_employment VARCHAR, 
    family_support FLOAT, 
    living_arrangement VARCHAR, 
    employment_status VARCHAR, 
    financial_stress BOOLEAN, 
    communication_skills FLOAT, 
    social_connection FLOAT, 
    sleep_pattern VARCHAR, 
    exercise_frequency VARCHAR, 
    substance_use VARCHAR, 
    profile_score FLOAT, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    updated_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_profile_assessments_id ON profile_assessments (id);

CREATE TABLE resources (
    id SERIAL NOT NULL, 
    title VARCHAR, 
    category VARCHAR, 
    description TEXT, 
    url VARCHAR, 
    phone VARCHAR, 
    is_active BOOLEAN, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_resources_id ON resources (id);

CREATE TABLE alerts (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    alert_type VARCHAR, 
    risk_level VARCHAR, 
    message TEXT, 
    is_read BOOLEAN, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_alerts_created_at ON alerts (created_at);

CREATE INDEX ix_alerts_id ON alerts (id);

CREATE INDEX ix_alerts_is_read ON alerts (is_read);

CREATE INDEX ix_alerts_risk_level ON alerts (risk_level);

CREATE TABLE chat_messages (
    id SERIAL NOT NULL, 
    sender_id INTEGER, 
    receiver_id INTEGER, 
    message TEXT, 
    message_type VARCHAR, 
    is_read BOOLEAN, 
    read_at TIMESTAMP WITHOUT TIME ZONE, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    updated_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(receiver_id) REFERENCES users (id), 
    FOREIGN KEY(sender_id) REFERENCES users (id)
);

CREATE INDEX ix_chat_messages_id ON chat_messages (id);

CREATE TABLE counselor_sessions (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    counselor_id INTEGER, 
    session_type VARCHAR, 
    status VARCHAR, 
    risk_level_at_escalation VARCHAR, 
    counselor_notes TEXT, 
    intervention_type VARCHAR, 
    outcome VARCHAR, 
    follow_up_needed BOOLEAN, 
    follow_up_date TIMESTAMP WITHOUT TIME ZONE, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    updated_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(counselor_id) REFERENCES users (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_counselor_sessions_id ON counselor_sessions (id);

CREATE TABLE modality_predictions (
    id SERIAL NOT NULL, 
    student_id INTEGER NOT NULL, 
    modality VARCHAR NOT NULL, 
    source_type VARCHAR NOT NULL, 
    source_record_id INTEGER NOT NULL, 
    model_registry_id INTEGER NOT NULL, 
    predicted_class VARCHAR NOT NULL, 
    probability FLOAT NOT NULL, 
    score_0_100 FLOAT NOT NULL, 
    confidence FLOAT NOT NULL, 
    explanation_json JSON, 
    processing_time_ms FLOAT, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_modality_predictions_confidence_0_1 CHECK (confidence >= 0 AND confidence <= 1), 
    CONSTRAINT ck_modality_predictions_probability_0_1 CHECK (probability >= 0 AND probability <= 1), 
    CONSTRAINT ck_modality_predictions_score_0_100 CHECK (score_0_100 >= 0 AND score_0_100 <= 100), 
    FOREIGN KEY(model_registry_id) REFERENCES model_registry (id) ON DELETE RESTRICT, 
    FOREIGN KEY(student_id) REFERENCES users (id)
);

CREATE INDEX ix_modality_predictions_created_at ON modality_predictions (created_at);

CREATE INDEX ix_modality_predictions_id ON modality_predictions (id);

CREATE INDEX ix_modality_predictions_modality ON modality_predictions (modality);

CREATE INDEX ix_modality_predictions_source ON modality_predictions (source_type, source_record_id);

CREATE INDEX ix_modality_predictions_student_created ON modality_predictions (student_id, created_at);

CREATE TABLE risk_assessments (
    id SERIAL NOT NULL, 
    student_id INTEGER NOT NULL, 
    fusion_model_id INTEGER, 
    final_probability FLOAT NOT NULL, 
    final_score FLOAT NOT NULL, 
    risk_level VARCHAR NOT NULL, 
    confidence FLOAT NOT NULL, 
    data_completeness JSON, 
    safety_override BOOLEAN NOT NULL, 
    safety_override_reason TEXT, 
    explanation_json JSON, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_risk_assessments_confidence_0_1 CHECK (confidence >= 0 AND confidence <= 1), 
    CONSTRAINT ck_risk_assessments_final_probability_0_1 CHECK (final_probability >= 0 AND final_probability <= 1), 
    CONSTRAINT ck_risk_assessments_final_score_0_100 CHECK (final_score >= 0 AND final_score <= 100), 
    FOREIGN KEY(fusion_model_id) REFERENCES model_registry (id) ON DELETE RESTRICT, 
    FOREIGN KEY(student_id) REFERENCES users (id)
);

CREATE INDEX ix_risk_assessments_created_at ON risk_assessments (created_at);

CREATE INDEX ix_risk_assessments_id ON risk_assessments (id);

CREATE INDEX ix_risk_assessments_risk_level ON risk_assessments (risk_level);

CREATE INDEX ix_risk_assessments_safety_override ON risk_assessments (safety_override);

CREATE INDEX ix_risk_assessments_student_created ON risk_assessments (student_id, created_at);

CREATE TABLE safetalk_bot_messages (
    id SERIAL NOT NULL, 
    user_id INTEGER, 
    user_message TEXT, 
    bot_response TEXT, 
    intent VARCHAR, 
    confidence FLOAT, 
    crisis_level INTEGER, 
    response_details JSON, 
    created_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(user_id) REFERENCES users (id)
);

CREATE INDEX ix_safetalk_bot_messages_id ON safetalk_bot_messages (id);

CREATE TABLE worker_jobs (
    id SERIAL NOT NULL, 
    job_type VARCHAR NOT NULL, 
    source_type VARCHAR NOT NULL, 
    source_record_id INTEGER NOT NULL, 
    status VARCHAR NOT NULL, 
    attempts INTEGER NOT NULL, 
    error_message TEXT, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    started_at TIMESTAMP WITHOUT TIME ZONE, 
    completed_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_worker_jobs_created_at ON worker_jobs (created_at);

CREATE INDEX ix_worker_jobs_id ON worker_jobs (id);

CREATE INDEX ix_worker_jobs_source ON worker_jobs (source_type, source_record_id);

CREATE INDEX ix_worker_jobs_status ON worker_jobs (status);

CREATE TABLE alert_events (
    id SERIAL NOT NULL, 
    alert_id INTEGER NOT NULL, 
    old_status VARCHAR, 
    new_status VARCHAR NOT NULL, 
    changed_by_user_id INTEGER, 
    notes TEXT, 
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(alert_id) REFERENCES alerts (id), 
    FOREIGN KEY(changed_by_user_id) REFERENCES users (id)
);

CREATE INDEX ix_alert_events_alert_created ON alert_events (alert_id, created_at);

CREATE INDEX ix_alert_events_created_at ON alert_events (created_at);

CREATE INDEX ix_alert_events_id ON alert_events (id);

CREATE INDEX ix_alert_events_new_status ON alert_events (new_status);

CREATE TABLE risk_assessment_inputs (
    id SERIAL NOT NULL, 
    risk_assessment_id INTEGER NOT NULL, 
    modality_prediction_id INTEGER NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(modality_prediction_id) REFERENCES modality_predictions (id), 
    FOREIGN KEY(risk_assessment_id) REFERENCES risk_assessments (id), 
    CONSTRAINT uq_risk_assessment_input_pair UNIQUE (risk_assessment_id, modality_prediction_id)
);

CREATE INDEX ix_risk_assessment_inputs_id ON risk_assessment_inputs (id);

CREATE INDEX ix_risk_assessment_inputs_assessment ON risk_assessment_inputs (risk_assessment_id);

CREATE INDEX ix_risk_assessment_inputs_prediction ON risk_assessment_inputs (modality_prediction_id);

INSERT INTO alembic_version (version_num) VALUES ('20260714_0001') RETURNING alembic_version.version_num;

COMMIT;


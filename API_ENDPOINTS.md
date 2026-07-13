# Suicide Prevention Agent API - Endpoints Reference

## Base URL
```
http://localhost:8000
```

## API Documentation
- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`

---

## Authentication Endpoints (`/api/auth`)

### Register User
```
POST /api/auth/register
Content-Type: application/json

{
    "email": "user@university.edu",
    "username": "username",
    "full_name": "Full Name",
    "password": "SecurePassword123!",
    "role": "student",
    "department": "Computer Science",
    "year_of_study": 2
}

Response 200:
{
    "id": 1,
    "email": "user@university.edu",
    "username": "username",
    "full_name": "Full Name",
    "role": "student",
    "is_active": true,
    "created_at": "2024-03-11T10:00:00"
}
```

### Login
```
POST /api/auth/login
Content-Type: application/json

{
    "username": "username",
    "password": "SecurePassword123!"
}

Response 200:
{
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "user": { /* user object */ }
}
```

### Get Current User
```
GET /api/auth/me
Authorization: Bearer <access_token>

Response 200:
{ /* user object */ }
```

### Update Profile
```
PUT /api/auth/me
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "full_name": "Updated Name",
    "department": "New Department",
    "year_of_study": 3
}

Response 200:
{ /* updated user object */ }
```

### Verify Token
```
GET /api/auth/verify/{token}

Response 200:
{
    "valid": true,
    "username": "username"
}
```

### Logout
```
POST /api/auth/logout
Authorization: Bearer <access_token>

Response 200:
{
    "message": "Successfully logged out"
}
```

---

## Assessment Endpoints (`/api/assessments`)

### Create Profile Assessment
```
POST /api/assessments/profile
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "user_id": 1,
    "gpa": 3.5,
    "repeated_subjects": 1,
    "attendance": 95,
    "family_relationship_score": 8,
    "income_level": "Medium",
    "living_arrangement": "With Family",
    "employment_status": "Student",
    "financial_stress": false,
    "communication_skills": 7,
    "social_connection": 7,
    "sleep_pattern": "Regular",
    "exercise_frequency": "Regularly",
    "substance_use": "None"
}

Response 200:
{
    "id": 1,
    "user_id": 1,
    "profile_score": 15.5,
    "created_at": "2024-03-11T10:00:00"
}
```

### Create DASS21 Assessment
```
POST /api/assessments/dass21
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "responses": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]
}

Response 200:
{
    "id": 1,
    "user_id": 1,
    "dass21_data": {
        "depression_score": 14,
        "anxiety_score": 8,
        "stress_score": 12,
        "total_dass21_score": 34,
        "depression_severity": "Mild",
        "anxiety_severity": "Mild",
        "stress_severity": "Mild"
    },
    "dass21_risk_score": 26.98,
    "created_at": "2024-03-11T10:00:00"
}
```

### Get Latest DASS21
```
GET /api/assessments/dass21/latest
Authorization: Bearer <access_token>

Response 200:
{
    "id": 1,
    "depression_score": 14,
    "anxiety_score": 8,
    "stress_score": 12,
    "total_dass21_score": 34,
    "created_at": "2024-03-11T10:00:00"
}
```

### Multimodal Risk Assessment
```
POST /api/assessments/risk-assessment
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "user_id": 1,
    "scores": {
        "profile_score": 15.5,
        "mood_score": 25,
        "dass21_score": 26.98,
        "text_score": 30,
        "voice_score": 20,
        "face_score": 35,
        "behavioral_score": 20
    },
    "profile_data": {
        "living_arrangement": "Alone",
        "financial_stress": true
    }
}

Response 200:
{
    "user_id": 1,
    "composite_score": 43.2,
    "risk_level": "MEDIUM",
    "needs_escalation": false,
    "recommendations": [
        "Schedule a check-in with university counselor",
        "Try guided breathing exercises",
        "Join a student support group"
    ],
    "timestamp": "2024-03-11T10:00:00",
    "modality_breakdown": { /* scores object */ }
}
```

### Get Assessment History
```
GET /api/assessments/history/{user_id}?limit=20&days=30
Authorization: Bearer <access_token>

Response 200:
{
    "user_id": 1,
    "date_range": {
        "start": "2024-02-10T10:00:00",
        "end": "2024-03-11T10:00:00"
    },
    "total_assessments": 5,
    "assessments": [
        {
            "id": 5,
            "assessment_type": "multimodal",
            "composite_score": 43.2,
            "risk_level": "MEDIUM",
            "created_at": "2024-03-11T10:00:00"
        }
    ]
}
```

---

## Daily Check-in Endpoints (`/api/checkin`)

### Create Daily Check-in
```
POST /api/checkin/today
Authorization: Bearer <access_token>
Content-Type: application/json

{
    "mood": 3,
    "mood_description": "Feeling okay but stressed",
    "sleep_hours": 7.5,
    "exercise_minutes": 30,
    "social_interaction": "Moderate",
    "stress_level": 6,
    "anxiety_level": 5,
    "negative_thoughts": false,
    "substance_use_today": false,
    "self_harm_thoughts": false,
    "notes": "Had a good day at university"
}

Response 200:
{
    "id": 1,
    "user_id": 1,
    "checkin_risk_score": 52.5,
    "mood": 3,
    "sleep_hours": 7.5,
    "stress_level": 6,
    "created_at": "2024-03-11T10:00:00",
    "message": "Daily check-in recorded successfully"
}
```

### Get Today's Check-in
```
GET /api/checkin/today
Authorization: Bearer <access_token>

Response 200:
{
    "checked_in": true,
    "id": 1,
    "mood": 3,
    "sleep_hours": 7.5,
    "stress_level": 6,
    "daily_risk_score": 52.5,
    "created_at": "2024-03-11T10:00:00"
}

Response (not checked in):
{
    "checked_in": false,
    "message": "You haven't checked in today yet"
}
```

### Get Check-in History
```
GET /api/checkin/history?limit=30&days=90
Authorization: Bearer <access_token>

Response 200:
{
    "user_id": 1,
    "total_checkins": 15,
    "history": [
        {
            "id": 15,
            "mood": 3,
            "stress_level": 6,
            "daily_risk_score": 52.5,
            "self_harm_thoughts": false,
            "created_at": "2024-03-11T10:00:00"
        }
    ]
}
```

### Get Check-in Statistics
```
GET /api/checkin/stats?days=30
Authorization: Bearer <access_token>

Response 200:
{
    "user_id": 1,
    "total_checkins": 15,
    "average_mood": 3.2,
    "average_stress": 5.8,
    "average_anxiety": 5.1,
    "average_sleep_hours": 7.3,
    "self_harm_thoughts_count": 0,
    "negative_thoughts_count": 2,
    "substance_use_count": 0
}
```

---

## Counselor Endpoints (`/api/counselor`)

### Get Alerts
```
GET /api/counselor/alerts?unread_only=false&limit=50
Authorization: Bearer <access_token>
Role: counselor, admin

Response 200:
{
    "counselor_id": 2,
    "total_alerts": 5,
    "alerts": [
        {
            "id": 1,
            "user_id": 1,
            "alert_type": "escalation",
            "risk_level": "HIGH",
            "message": "User has HIGH risk level...",
            "is_read": false,
            "created_at": "2024-03-11T09:00:00"
        }
    ]
}
```

### Get High-Risk Users
```
GET /api/counselor/high-risk-users?risk_level=HIGH&limit=50
Authorization: Bearer <access_token>
Role: counselor, admin

Response 200:
{
    "counselor_id": 2,
    "total_users": 3,
    "high_risk_users": [
        {
            "user_id": 1,
            "name": "John Doe",
            "email": "john@university.edu",
            "risk_level": "HIGH",
            "composite_score": 75.2,
            "last_assessment": "2024-03-11T09:00:00"
        }
    ]
}
```

### Get Student Dashboard
```
GET /api/counselor/student/{user_id}/dashboard
Authorization: Bearer <access_token>
Role: counselor, admin

Response 200:
{
    "user": {
        "id": 1,
        "name": "John Doe",
        "email": "john@university.edu",
        "department": "Computer Science",
        "year_of_study": 2
    },
    "profile_assessment": {
        "profile_score": 15.5,
        "updated_at": "2024-03-10T10:00:00"
    },
    "dass21_assessment": {
        "depression_score": 14,
        "anxiety_score": 8,
        "total_dass21_score": 34,
        "created_at": "2024-03-11T09:00:00"
    },
    "today_checkin": {
        "mood": 3,
        "stress_level": 6,
        "self_harm_thoughts": false,
        "created_at": "2024-03-11T10:00:00"
    },
    "latest_risk_assessment": {
        "composite_score": 43.2,
        "risk_level": "MEDIUM",
        "needs_escalation": false,
        "created_at": "2024-03-11T10:00:00"
    },
    "critical_alerts": {
        "self_harm_thoughts": 0,
        "negative_thoughts": 0,
        "substance_use": 0
    }
}
```

### Create Counselor Session
```
POST /api/counselor/sessions
Authorization: Bearer <access_token>
Role: counselor, admin
Content-Type: application/json

{
    "user_id": 1,
    "session_type": "auto_escalated",
    "risk_level_at_escalation": "HIGH",
    "counselor_notes": "Student showing signs of distress"
}

Response 200:
{
    "id": 1,
    "user_id": 1,
    "counselor_id": 2,
    "status": "in_progress",
    "created_at": "2024-03-11T10:00:00"
}
```

### Update Counselor Session
```
PUT /api/counselor/sessions/{session_id}
Authorization: Bearer <access_token>
Role: counselor, admin
Content-Type: application/json

{
    "status": "completed",
    "counselor_notes": "Updated counselor notes",
    "intervention_type": "cognitive_behavioral",
    "outcome": "positive",
    "follow_up_needed": true,
    "follow_up_date": "2024-03-18T10:00:00"
}

Response 200:
{
    "message": "Session updated successfully",
    "session_id": 1
}
```

### Get Analytics Summary
```
GET /api/counselor/analytics/summary?days=30
Authorization: Bearer <access_token>
Role: counselor, admin

Response 200:
{
    "period_days": 30,
    "total_students": 150,
    "at_risk_students": 12,
    "risk_percentage": 8.0,
    "high_risk_assessments": 15,
    "risk_distribution": {
        "HIGH": 10,
        "SEVERE": 5
    },
    "counselor_sessions": 8,
    "sessions_completed": 6
}
```

---

## Resources Endpoints (`/api/resources`)

### Get All Resources
```
GET /api/resources/?category=crisis
Authorization: Bearer <access_token>

Response 200:
{
    "total_resources": 5,
    "resources": [
        {
            "id": 1,
            "title": "National Crisis Hotline",
            "category": "crisis",
            "description": "24/7 crisis support",
            "phone": "1-800-273-8255"
        }
    ]
}
```

### Get Crisis Resources
```
GET /api/resources/crisis
Authorization: Bearer <access_token>

Response 200:
{
    "crisis_resources": [
        {
            "id": 1,
            "title": "National Suicide Prevention Lifeline",
            "description": "24/7 free and confidential support",
            "phone": "1-800-273-8255"
        }
    ]
}
```

---

## Error Responses

### 400 Bad Request
```json
{
    "error": "Invalid input data",
    "status_code": 400,
    "timestamp": "2024-03-11T10:00:00"
}
```

### 401 Unauthorized
```json
{
    "error": "Invalid authentication credentials",
    "status_code": 401,
    "timestamp": "2024-03-11T10:00:00"
}
```

### 403 Forbidden
```json
{
    "error": "Only admins can create resources",
    "status_code": 403,
    "timestamp": "2024-03-11T10:00:00"
}
```

### 404 Not Found
```json
{
    "error": "User not found",
    "status_code": 404,
    "timestamp": "2024-03-11T10:00:00"
}
```

### 500 Internal Server Error
```json
{
    "error": "Internal server error",
    "status_code": 500,
    "timestamp": "2024-03-11T10:00:00"
}
```

---

## Authentication

All endpoints except registration and login require JWT authentication:

```
Authorization: Bearer <access_token>
```

Token is obtained from login endpoint and expires based on `ACCESS_TOKEN_EXPIRE_MINUTES` setting.

## Rate Limiting

Currently no rate limiting is implemented. Consider adding:
- Per-user rate limits
- Per-endpoint rate limits
- IP-based rate limiting

## Notes

- All timestamps are in ISO 8601 format
- User roles: `student`, `counselor`, `admin`, `psychiatrist`
- Risk levels: `LOW`, `MEDIUM`, `HIGH`, `SEVERE`
- Mood scale: 1-5 (1 = very sad, 5 = very happy)
- Stress/Anxiety scale: 1-10 (1 = none, 10 = extreme)

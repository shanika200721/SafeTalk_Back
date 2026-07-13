# Sample Data Guide - Database Seeding

## Overview

The enhanced `seed_student_data.py` script creates a comprehensive set of sample data for testing the suicide prevention chatbot system. This includes multiple students with complete profiles, DASS21 assessments, and 30 days of daily check-in data.

---

## Test Students Created

### 1. **Test Student** (Low Risk)
- **Email**: student@example.com
- **Username**: student
- **Password**: Student123!
- **Department**: Computer Science
- **Year**: 2nd Year
- **Profile Score**: 3.5 GPA, 95% Attendance
- **DASS21 Scores**:
  - Depression: 2 (Normal)
  - Anxiety: 2 (Normal)
  - Stress: 2 (Normal)
  - Total: 6/126 (Very Low Risk)
- **Risk Level**: LOW
- **Daily Check-ins**: 30 entries (7-30 days ago)

---

### 2. **Alex Miller** (Medium Risk)
- **Email**: alex.miller@example.com
- **Username**: alexmiller
- **Password**: Student123!
- **Department**: Business
- **Year**: 3rd Year
- **Profile Score**: 2.8 GPA, 75% Attendance, Financial Stress
- **DASS21 Scores**:
  - Depression: 28 (Moderate)
  - Anxiety: 28 (Moderate)
  - Stress: 24 (Mild)
  - Total: 80/126 (Moderate Risk)
- **Risk Level**: MEDIUM
- **Daily Check-ins**: 30 entries with varied stress/anxiety levels
- **Triggers**: Financial stress, low attendance

---

### 3. **Jordan Smith** (Medium Risk)
- **Email**: jordan.smith@example.com
- **Username**: jordansmith
- **Password**: Student123!
- **Department**: Engineering
- **Year**: 1st Year
- **Profile Score**: 3.2 GPA, 88% Attendance
- **DASS21 Scores**:
  - Depression: 24 (Mild)
  - Anxiety: 26 (Moderate)
  - Stress: 22 (Mild)
  - Total: 72/126 (Moderate Risk)
- **Risk Level**: MEDIUM
- **Daily Check-ins**: 30 entries with fluctuating anxiety
- **Triggers**: Academic pressure, financial stress

---

### 4. **Casey Johnson** (Low Risk)
- **Email**: casey.johnson@example.com
- **Username**: caseyjohnson
- **Password**: Student123!
- **Department**: Psychology
- **Year**: 4th Year
- **Profile Score**: 3.8 GPA, 100% Attendance
- **DASS21 Scores**:
  - Depression: 2 (Normal)
  - Anxiety: 0 (Normal)
  - Stress: 2 (Normal)
  - Total: 4/126 (Minimal Risk)
- **Risk Level**: LOW
- **Daily Check-ins**: 30 entries with consistently positive mood
- **Triggers**: None significant

---

### 5. **Taylor Brown** (High Risk)
- **Email**: taylor.brown@example.com
- **Username**: taylorbrown
- **Password**: Student123!
- **Department**: Medicine
- **Year**: 2nd Year
- **Profile Score**: 2.5 GPA, 65% Attendance
- **DASS21 Scores**:
  - Depression: 42 (Extremely Severe)
  - Anxiety: 42 (Extremely Severe)
  - Stress: 42 (Extremely Severe)
  - Total: 126/126 (Severe Risk)
- **Risk Level**: HIGH
- **Daily Check-ins**: 30 entries with severe mood variations
- **Triggers**: Multiple stressors including financial, academic, family issues

---

## Database Schema - Sample Data Structure

### Users Table
```
id | email | username | full_name | department | year_of_study | role | is_active | created_at
```

### Profile Assessments Table
Each student has ONE profile assessment containing:
- Academic metrics (GPA, attendance, communication skills)
- Family/Support information
- Financial status
- Exercise frequency
- Calculated profile risk score

### DASS21 Assessments Table
Each student has ONE DASS21 assessment containing:
- **21 Response Items** (0-3 scale each):
  - Items 1-7: Depression indicators
  - Items 8-14: Anxiety indicators
  - Items 15-21: Stress indicators
- **Calculated Scores**:
  - Depression (0-42)
  - Anxiety (0-42)
  - Stress (0-42)
  - Total (0-126)
- **Severity Classifications**: Normal, Mild, Moderate, Severe, Extremely Severe

### Daily Check-ins Table
Each student has **30 historical entries** (1 per day for last 30 days):
- **Daily Metrics**:
  - Mood (1-5 scale)
  - Stress level (1-10)
  - Anxiety level (1-10)
  - Sleep hours (5.5-9.5)
  - Exercise minutes (0-60)
  - Social interaction quality
  - Negative thoughts (boolean)
  - Substance use (boolean)
- **Timestamp**: Created with historical dates (30 days ago to today)
- **Notes**: Context-specific journal entry notes

### Assessments Table
Each student has ONE overall multimodal assessment:
- Profile score component
- Mood score component
- DASS21 score component (normalized 0-100)
- Composite risk score
- Overall risk level (LOW/MEDIUM/HIGH/SEVERE)
- Escalation flag
- Recommendations array

---

## DASS21 Scoring Reference

### Raw Score to Final Score
DASS21 uses a 0-3 scale (0=Did not apply, 3=Applied most of the time):
1. Sum responses for each subscale (7 items each)
2. Multiply by 2 for final score

**Final Score Ranges:**
- **Depression**: 0-42
  - 0-9: Normal
  - 10-13: Mild
  - 14-20: Moderate
  - 21-27: Severe
  - 28+: Extremely Severe

- **Anxiety**: 0-42
  - 0-7: Normal
  - 8-9: Mild
  - 10-14: Moderate
  - 15-19: Severe
  - 20+: Extremely Severe

- **Stress**: 0-42
  - 0-14: Normal
  - 15-18: Mild
  - 19-25: Moderate
  - 26-33: Severe
  - 34+: Extremely Severe

---

## How to Run Seeding Script

### Prerequisites
```bash
cd backend
pip install -r requirements.txt
```

### Execute Seeding
```bash
python seed_student_data.py
```

### Expected Output
```
🌱 Seeding comprehensive student test data...
============================================================

📝 Creating student: Test Student
  📊 Creating 30 days of daily check-ins...
  ✅ Test Student created successfully!
     DASS21: Depression=2, Anxiety=2, Stress=2
     Risk Level: LOW
     Daily Check-ins: 30 entries created

[... 4 more students ...]

============================================================
🎉 Database seeding completed successfully!

📋 Test Credentials:
...
```

---

## Accessing Sample Data

### Via REST API

#### Get Student Profile
```bash
GET /api/student/profile
```

#### Get DASS21 Assessment
```bash
GET /api/assessment/dass21
```

#### Get Daily Check-ins (Last 30 Days)
```bash
GET /api/checkin/history?days=30
```

#### Get Risk Assessment
```bash
GET /api/assessment/composite
```

---

## Data Reset & Cleanup

### Clear All Seeded Data
```bash
python -c "
from app.database import SessionLocal
from app.models.database_models import Base, engine
Base.metadata.drop_all(engine)
print('✅ Database cleared')
"
```

### Re-seed Fresh Data
```bash
# Clear first
python -c "from app.models.database_models import Base, engine; Base.metadata.drop_all(engine)"
# Then seed
python seed_student_data.py
```

---

## Testing Scenarios

### Test Case 1: Low-Risk Student Dashboard
1. Login as: student@example.com / Student123!
2. View stable mood patterns
3. Review normal DASS21 scores
4. Verify NO escalation triggers

### Test Case 2: Medium-Risk Student Monitoring
1. Login as: alex.miller@example.com / Student123!
2. View fluctuating stress/anxiety
3. Review moderate DASS21 scores
4. Verify counselor alerts appear

### Test Case 3: High-Risk Student Alert System
1. Login as: taylor.brown@example.com / Student123!
2. View severe mood variations
3. Review extremely high DASS21 scores
4. Verify immediate escalation to counselor
5. Test crisis intervention protocols

### Test Case 4: 30-Day Historical Analysis
1. Select any student
2. View full daily check-in history
3. Verify trend analysis across 30 days
4. Test data visualization/charting

---

## Key Features Tested

✅ **DASS21 Assessment**
- 21-item questionnaire with proper scoring
- Severity classification system
- Longitudinal tracking capability

✅ **Daily Check-ins**
- 30 days of historical data per student
- Varied mood patterns per risk level
- Correlations with DASS21 scores

✅ **Risk Assessment**
- Multi-factor risk calculation
- Proper escalation triggers
- Tailored recommendations

✅ **Data Relationships**
- Proper database relationships
- Foreign key constraints
- Data integrity validation

---

## Extending Sample Data

### Add New Student
Edit `STUDENTS_DATA` in `seed_student_data.py`:
```python
{
    "email": "newemail@example.com",
    "username": "newusername",
    "full_name": "New Student",
    "department": "Department",
    "year_of_study": 1,
    "profile": { ... },
    "dass21_responses": [0, 1, 0, ...],  # 21 items
    "risk_level": "LOW|MEDIUM|HIGH"
}
```

### Modify DASS21 Responses
Change values in `dass21_responses` array (21 items, 0-3 scale each)

### Adjust Daily Check-in Patterns
Modify the loop in `create_student_with_data()` function

---

## Data Validation

All seeded data includes:
- ✅ Valid email formats
- ✅ Proper DASS21 scoring (21 items × 0-3 scale)
- ✅ Consistent severity classifications
- ✅ Historical timestamp validation
- ✅ Foreign key referential integrity
- ✅ Appropriate risk level assignments

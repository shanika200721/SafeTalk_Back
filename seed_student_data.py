"""
Seed database with comprehensive test student data including DASS21 and Daily Check-ins
Also populates data for any existing students already registered
"""

import sys
sys.path.insert(0, '/c/Users/CMSDEV04/Documents/research/suicide-prevention-agent/backend')

from app.database import SessionLocal
from app.models.database_models import (
    User, ProfileAssessment, DASS21Assessment, 
    DailyCheckIn, Assessment, UserRole
)
from datetime import datetime, timedelta
from argon2 import PasswordHasher
from sqlalchemy import func
import random

ph = PasswordHasher()
db = SessionLocal()

# Existing student data loader
def populate_existing_students_data():
    """Populate data for any existing students already in the database"""
    print("\n📝 Checking for existing students...")
    
    # Get all students who don't have DASS21 assessments yet
    existing_students = db.query(User).filter(User.role == UserRole.STUDENT).all()
    
    if not existing_students:
        print("   ℹ️ No existing students found to populate")
        return 0
    
    populated_count = 0
    
    for student in existing_students:
        # Check if student already has DASS21
        existing_dass21 = db.query(DASS21Assessment).filter(
            DASS21Assessment.user_id == student.id
        ).first()
        
        if existing_dass21:
            print(f"   ⏭️  {student.full_name} - already has DASS21, skipping...")
            continue
        
        print(f"   📊 Populating data for: {student.full_name}")
        
        # Generate random but realistic DASS21 responses
        dass21_responses = [random.randint(0, 3) for _ in range(21)]
        dass21_data = calculate_dass21_scores(dass21_responses)
        
        # Create DASS21 assessment
        dass21 = DASS21Assessment(
            user_id=student.id,
            responses=dass21_responses,
            depression_score=dass21_data["depression_score"],
            anxiety_score=dass21_data["anxiety_score"],
            stress_score=dass21_data["stress_score"],
            total_dass21_score=dass21_data["total_dass21_score"],
            depression_severity=dass21_data["depression_severity"],
            anxiety_severity=dass21_data["anxiety_severity"],
            stress_severity=dass21_data["stress_severity"],
            created_at=datetime.now() - timedelta(days=random.randint(1, 30))
        )
        db.add(dass21)
        db.flush()
        
        # Create 30 days of daily check-ins if not already created
        existing_checkins = db.query(DailyCheckIn).filter(
            DailyCheckIn.user_id == student.id
        ).count()
        
        if existing_checkins == 0:
            print(f"      📅 Creating 30 daily check-ins...")
            base_stress = dass21_data["stress_score"] / 10
            base_anxiety = dass21_data["anxiety_score"] / 10
            
            for day in range(30, 0, -1):
                checkin_date = datetime.now() - timedelta(days=day)
                
                daily_checkin = DailyCheckIn(
                    user_id=student.id,
                    mood=max(1, min(5, 5 - int(dass21_data["depression_score"] / 10))),
                    mood_description=["Very Poor", "Poor", "Fair", "Good", "Great"][
                        max(0, min(4, 4 - int(dass21_data["depression_score"] / 10)))
                    ],
                    stress_level=int(max(1, min(10, base_stress + random.uniform(-2, 2)))),
                    anxiety_level=int(max(1, min(10, base_anxiety + random.uniform(-2, 2)))),
                    sleep_hours=round(random.uniform(5.5, 9.5), 1),
                    exercise_minutes=random.choice([0, 0, 15, 20, 30, 45, 60]),
                    social_interaction=random.choice(["None", "Limited", "Moderate", "Good", "Excellent"]),
                    negative_thoughts=random.choice([False, False, False, True]) if dass21_data["depression_score"] > 20 else random.choice([False, False, False, False, True]),
                    substance_use_today=random.choice([False, False, False, False, False, True]) if dass21_data["total_dass21_score"] > 90 else False,
                    self_harm_thoughts=False,
                    notes=f"Check-in: {checkin_date.strftime('%Y-%m-%d')}",
                    created_at=checkin_date
                )
                db.add(daily_checkin)
            db.flush()
        
        # Create overall assessment if not exists
        existing_assessment = db.query(Assessment).filter(
            Assessment.user_id == student.id,
            Assessment.assessment_type == "multimodal"
        ).first()
        
        if not existing_assessment:
            dass21_score_normalized = (dass21_data["total_dass21_score"] / 126) * 100
            
            # Determine risk level based on DASS21
            if dass21_data["total_dass21_score"] < 20:
                risk_level = "LOW"
            elif dass21_data["total_dass21_score"] < 60:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            assessment = Assessment(
                user_id=student.id,
                assessment_type="multimodal",
                profile_score=random.uniform(20, 80),
                mood_score=random.uniform(30, 80),
                dass21_score=dass21_score_normalized,
                composite_score=random.uniform(20, 80),
                risk_level=risk_level,
                needs_escalation=risk_level in ["HIGH", "SEVERE"],
                recommendations=["Attend counseling sessions", "Practice stress management"] if risk_level != "LOW" else ["Continue wellness routine"]
            )
            db.add(assessment)
            db.flush()
        
        db.commit()
        print(f"      ✅ {student.full_name} - Data populated successfully!")
        print(f"         DASS21: Depression={dass21_data['depression_score']}, Anxiety={dass21_data['anxiety_score']}, Stress={dass21_data['stress_score']}")
        populated_count += 1
    
    return populated_count

# Sample student data
STUDENTS_DATA = [
    {
        "email": "student@example.com",
        "username": "student",
        "full_name": "Test Student",
        "department": "Computer Science",
        "year_of_study": 2,
        "profile": {
            "gpa": 3.5,
            "attendance": 95,
            "family_relationship_score": 8,
            "communication_skills": 7,
            "family_support": 8,
            "financial_stress": False,
            "exercise_frequency": "3-4 times/week"
        },
        "dass21_responses": [0, 1, 0, 2, 1, 0, 1, 0, 0, 1, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1, 0],
        "risk_level": "LOW"
    },
    {
        "email": "alex.miller@example.com",
        "username": "alexmiller",
        "full_name": "Alex Miller",
        "department": "Business",
        "year_of_study": 3,
        "profile": {
            "gpa": 2.8,
            "attendance": 75,
            "family_relationship_score": 5,
            "communication_skills": 4,
            "family_support": 5,
            "financial_stress": True,
            "exercise_frequency": "1-2 times/week"
        },
        "dass21_responses": [2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1],
        "risk_level": "MEDIUM"
    },
    {
        "email": "jordan.smith@example.com",
        "username": "jordansmith",
        "full_name": "Jordan Smith",
        "department": "Engineering",
        "year_of_study": 1,
        "profile": {
            "gpa": 3.2,
            "attendance": 88,
            "family_relationship_score": 6,
            "communication_skills": 6,
            "family_support": 6,
            "financial_stress": True,
            "exercise_frequency": "2-3 times/week"
        },
        "dass21_responses": [1, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 1],
        "risk_level": "MEDIUM"
    },
    {
        "email": "casey.johnson@example.com",
        "username": "caseyjohnson",
        "full_name": "Casey Johnson",
        "department": "Psychology",
        "year_of_study": 4,
        "profile": {
            "gpa": 3.8,
            "attendance": 100,
            "family_relationship_score": 9,
            "communication_skills": 9,
            "family_support": 9,
            "financial_stress": False,
            "exercise_frequency": "Daily"
        },
        "dass21_responses": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "risk_level": "LOW"
    },
    {
        "email": "taylor.brown@example.com",
        "username": "taylorbrown",
        "full_name": "Taylor Brown",
        "department": "Medicine",
        "year_of_study": 2,
        "profile": {
            "gpa": 2.5,
            "attendance": 65,
            "family_relationship_score": 4,
            "communication_skills": 3,
            "family_support": 3,
            "financial_stress": True,
            "exercise_frequency": "Rarely"
        },
        "dass21_responses": [3, 3, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3],
        "risk_level": "HIGH"
    }
]

def calculate_dass21_scores(responses):
    """Calculate DASS21 scores from 21-item responses (0-3 scale)"""
    # Items 1-7: Depression (indices 0-6)
    depression_raw = sum(responses[i] for i in range(0, 7))
    # Items 8-14: Anxiety (indices 7-13)
    anxiety_raw = sum(responses[i] for i in range(7, 14))
    # Items 15-21: Stress (indices 14-21)
    stress_raw = sum(responses[i] for i in range(14, 21))
    
    # Multiply by 2 for final scores (DASS21 standard scoring)
    depression_score = depression_raw * 2
    anxiety_score = anxiety_raw * 2
    stress_score = stress_raw * 2
    total_score = depression_score + anxiety_score + stress_score
    
    # Determine severity levels
    def get_depression_severity(score):
        if score < 10: return "Normal"
        elif score < 14: return "Mild"
        elif score < 21: return "Moderate"
        elif score < 28: return "Severe"
        else: return "Extremely Severe"
    
    def get_anxiety_severity(score):
        if score < 8: return "Normal"
        elif score < 10: return "Mild"
        elif score < 15: return "Moderate"
        elif score < 20: return "Severe"
        else: return "Extremely Severe"
    
    def get_stress_severity(score):
        if score < 15: return "Normal"
        elif score < 19: return "Mild"
        elif score < 26: return "Moderate"
        elif score < 34: return "Severe"
        else: return "Extremely Severe"
    
    return {
        "depression_score": depression_score,
        "anxiety_score": anxiety_score,
        "stress_score": stress_score,
        "total_dass21_score": total_score,
        "depression_severity": get_depression_severity(depression_score),
        "anxiety_severity": get_anxiety_severity(anxiety_score),
        "stress_severity": get_stress_severity(stress_score)
    }

def create_student_with_data(student_data):
    """Create a complete student profile with DASS21 and daily check-ins"""
    
    # Check if student exists
    existing_student = db.query(User).filter(User.email == student_data["email"]).first()
    
    if existing_student:
        print(f"✅ Student already exists: {student_data['full_name']} (ID={existing_student.id})")
        return existing_student
    
    print(f"\n📝 Creating student: {student_data['full_name']}")
    
    # Create user
    user = User(
        email=student_data["email"],
        username=student_data["username"],
        full_name=student_data["full_name"],
        hashed_password=ph.hash("Student123!"),
        department=student_data["department"],
        year_of_study=student_data["year_of_study"],
        role=UserRole.STUDENT,
        is_active=True
    )
    db.add(user)
    db.flush()
    
    # Create profile assessment
    profile = ProfileAssessment(
        user_id=user.id,
        gpa=student_data["profile"]["gpa"],
        attendance=student_data["profile"]["attendance"],
        family_relationship_score=student_data["profile"]["family_relationship_score"],
        communication_skills=student_data["profile"]["communication_skills"],
        family_support=student_data["profile"]["family_support"],
        financial_stress=student_data["profile"]["financial_stress"],
        exercise_frequency=student_data["profile"]["exercise_frequency"],
        profile_score=random.uniform(20, 50)
    )
    db.add(profile)
    db.flush()
    
    # Create DASS21 assessment
    dass21_data = calculate_dass21_scores(student_data["dass21_responses"])
    dass21 = DASS21Assessment(
        user_id=user.id,
        responses=student_data["dass21_responses"],
        depression_score=dass21_data["depression_score"],
        anxiety_score=dass21_data["anxiety_score"],
        stress_score=dass21_data["stress_score"],
        total_dass21_score=dass21_data["total_dass21_score"],
        depression_severity=dass21_data["depression_severity"],
        anxiety_severity=dass21_data["anxiety_severity"],
        stress_severity=dass21_data["stress_severity"],
        created_at=datetime.now() - timedelta(days=random.randint(1, 30))
    )
    db.add(dass21)
    db.flush()
    
    # Create 30 days of daily check-ins
    print(f"  📊 Creating 30 days of daily check-ins...")
    for day in range(30, 0, -1):
        checkin_date = datetime.now() - timedelta(days=day)
        
        # Vary data based on DASS21 scores - higher scores = lower mood/higher anxiety
        base_stress = dass21_data["stress_score"] / 10  # Convert to 0-10 scale
        base_anxiety = dass21_data["anxiety_score"] / 10
        
        daily_checkin = DailyCheckIn(
            user_id=user.id,
            mood=max(1, min(5, 5 - int(dass21_data["depression_score"] / 10))),
            mood_description=["Very Poor", "Poor", "Fair", "Good", "Great"][
                max(0, min(4, 4 - int(dass21_data["depression_score"] / 10)))
            ],
            stress_level=int(max(1, min(10, base_stress + random.uniform(-2, 2)))),
            anxiety_level=int(max(1, min(10, base_anxiety + random.uniform(-2, 2)))),
            sleep_hours=round(random.uniform(5.5, 9.5), 1),
            exercise_minutes=random.choice([0, 0, 15, 20, 30, 45, 60]),
            social_interaction=random.choice(["None", "Limited", "Moderate", "Good", "Excellent"]),
            negative_thoughts=random.choice([False, False, False, True]) if dass21_data["depression_score"] > 20 else random.choice([False, False, False, False, True]),
            substance_use_today=random.choice([False, False, False, False, False, True]) if student_data["risk_level"] == "HIGH" else False,
            self_harm_thoughts=False,
            notes=f"Daily check-in: {checkin_date.strftime('%Y-%m-%d %H:%M')}. Feeling {['terrible', 'bad', 'okay', 'good', 'great'][max(1, min(5, 5 - int(dass21_data['depression_score'] / 10))) - 1].lower()}.",
            created_at=checkin_date
        )
        db.add(daily_checkin)
    
    # Create overall assessment
    dass21_score_normalized = (dass21_data["total_dass21_score"] / 126) * 100  # Max DASS21 is 126
    assessment = Assessment(
        user_id=user.id,
        assessment_type="multimodal",
        profile_score=profile.profile_score,
        mood_score=random.uniform(30, 80),
        dass21_score=dass21_score_normalized,
        composite_score=random.uniform(20, 80),
        risk_level=student_data["risk_level"],
        needs_escalation=student_data["risk_level"] in ["HIGH", "SEVERE"],
        recommendations=[
            "Attend counseling sessions",
            "Practice stress management",
            "Maintain regular exercise routine"
        ] if student_data["risk_level"] != "LOW" else ["Continue current wellness routine"]
    )
    db.add(assessment)
    
    db.commit()
    
    print(f"  ✅ {student_data['full_name']} created successfully!")
    print(f"     DASS21: Depression={dass21_data['depression_score']}, Anxiety={dass21_data['anxiety_score']}, Stress={dass21_data['stress_score']}")
    print(f"     Risk Level: {student_data['risk_level']}")
    print(f"     Daily Check-ins: 30 entries created")
    
    return user

try:
    print("🌱 Seeding comprehensive student data...")
    print("=" * 70)
    
    # STEP 1: Populate data for existing students
    print("\n🔍 STEP 1: Populating data for existing students")
    print("-" * 70)
    existing_populated = populate_existing_students_data()
    
    # STEP 2: Create test students
    print("\n\n👥 STEP 2: Creating test students")
    print("-" * 70)
    students = []
    for student_data in STUDENTS_DATA:
        student = create_student_with_data(student_data)
        students.append(student)
    
    print("\n" + "=" * 70)
    print("🎉 Database seeding completed successfully!")
    
    print("\n📊 SUMMARY:")
    print("-" * 70)
    print(f"✅ Existing students populated with data: {existing_populated}")
    print(f"✅ Test students created: {len(students)}")
    print(f"✅ Total DASS21 assessments: {existing_populated + len(students)}")
    print(f"✅ Total daily check-ins: {(existing_populated + len(students)) * 30}")
    
    print("\n📋 TEST CREDENTIALS:")
    print("-" * 70)
    for i, student_data in enumerate(STUDENTS_DATA):
        print(f"{i+1}. {student_data['full_name']}")
        print(f"   📧 Email: {student_data['email']}")
        print(f"   🔐 Password: Student123!")
        print(f"   ⚠️ Risk Level: {student_data['risk_level']}")
        print()
    
    print("=" * 70)
    
except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()

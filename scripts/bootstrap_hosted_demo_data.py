import argparse
import os
from datetime import datetime
from uuid import uuid4

from app.database import SessionLocal
from app.models.database_models import (
    ConsentRecord,
    CounselorAssignment,
    CounselorProfile,
    CounselorUniversityAssignment,
    University,
    User,
    UserRole,
)
from app.security import hash_password
from app.services.consent import CONSENT_TYPES, CURRENT_POLICY_VERSION


DEFAULT_UNIVERSITY = {
    "university_id": "uni-uok-main",
    "university_name": "University of Kelaniya",
    "university_code": "UOK",
    "campus_name": "Main Campus",
    "district": "Gampaha",
    "province": "Western",
    "email": "info@kln.ac.lk",
    "website": "https://www.kln.ac.lk",
}

DEFAULT_ADMIN_EMAIL = "admin@safetalk.app"


def public_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def get_or_create_university(db):
    university = (
        db.query(University)
        .filter(University.university_code == DEFAULT_UNIVERSITY["university_code"])
        .first()
    )
    if university:
        return university, False

    now = datetime.utcnow()
    university = University(**DEFAULT_UNIVERSITY, active=True, created_at=now, updated_at=now)
    db.add(university)
    db.flush()
    return university, True


def ensure_admin(db, password: str | None):
    admin = db.query(User).filter(User.username == "admin_001").first()
    created = False
    if not admin:
        admin = User(
            email=DEFAULT_ADMIN_EMAIL,
            username="admin_001",
            full_name="SafeTalk Administrator",
            role=UserRole.ADMIN,
            is_active=True,
        )
        db.add(admin)
        db.flush()
        created = True
    elif admin.email != DEFAULT_ADMIN_EMAIL:
        admin.email = DEFAULT_ADMIN_EMAIL

    if password:
        admin.hashed_password = hash_password(password)
        admin.is_active = True
        admin.updated_at = datetime.utcnow()

    return admin, created


def ensure_student_university(db, university):
    updated = 0
    students = db.query(User).filter(User.role == UserRole.STUDENT).all()
    for student in students:
        if student.university_id is None:
            student.university_id = university.id
            student.updated_at = datetime.utcnow()
            updated += 1
    return updated


def ensure_counselor_profiles(db, university):
    created = 0
    counselors = (
        db.query(User)
        .filter(User.role.in_([UserRole.COUNSELOR, UserRole.PSYCHIATRIST]))
        .order_by(User.id.asc())
        .all()
    )
    profiles = []
    for user in counselors:
        profile = db.query(CounselorProfile).filter(CounselorProfile.user_id == user.id).first()
        if not profile:
            profile = CounselorProfile(
                counselor_profile_id=public_id("cprof"),
                user_id=user.id,
                university_id=university.id,
                full_name=user.full_name or user.username,
                professional_title="Counselor" if user.role == UserRole.COUNSELOR else "Psychiatrist",
                email=user.email,
                available_days="Monday-Friday",
                available_from="09:00",
                available_until="17:00",
                availability_status="available",
                approved=True,
                student_visible=True,
                active=True,
            )
            db.add(profile)
            db.flush()
            created += 1
        else:
            profile.university_id = profile.university_id or university.id
            profile.approved = True
            profile.student_visible = True
            profile.active = True
            profile.updated_at = datetime.utcnow()
        profiles.append(profile)
    return profiles, created


def ensure_university_assignments(db, profiles, university, admin):
    created = 0
    for profile in profiles:
        existing = (
            db.query(CounselorUniversityAssignment)
            .filter(
                CounselorUniversityAssignment.counselor_profile_id == profile.id,
                CounselorUniversityAssignment.university_id == university.id,
                CounselorUniversityAssignment.active.is_(True),
            )
            .first()
        )
        if existing:
            continue
        db.add(
            CounselorUniversityAssignment(
                assignment_id=public_id("cuasg"),
                counselor_profile_id=profile.id,
                university_id=university.id,
                assigned_by=admin.id if admin else None,
                assignment_reason="Hosted demo bootstrap",
                active=True,
            )
        )
        created += 1
    return created


def ensure_student_assignments(db, profiles, admin):
    counselor_users = [profile.user for profile in profiles if profile.user and profile.user.is_active]
    if not counselor_users:
        return 0

    created = 0
    students = (
        db.query(User)
        .filter(User.role == UserRole.STUDENT, User.is_active.is_(True))
        .order_by(User.id.asc())
        .all()
    )
    for index, student in enumerate(students):
        existing = (
            db.query(CounselorAssignment)
            .filter(CounselorAssignment.student_id == student.id, CounselorAssignment.active.is_(True))
            .first()
        )
        if existing:
            continue
        counselor = counselor_users[index % len(counselor_users)]
        db.add(
            CounselorAssignment(
                assignment_id=public_id("asg"),
                student_id=student.id,
                counselor_id=counselor.id,
                assigned_by=admin.id if admin else None,
                assignment_reason="Hosted demo bootstrap",
                active=True,
            )
        )
        created += 1
    return created


def grant_demo_consents(db):
    created = 0
    students = db.query(User).filter(User.role == UserRole.STUDENT, User.is_active.is_(True)).all()
    for student in students:
        existing_types = {
            row.consent_type
            for row in db.query(ConsentRecord).filter(
                ConsentRecord.user_id == student.id,
                ConsentRecord.is_granted.is_(True),
                ConsentRecord.withdrawn_at.is_(None),
            )
        }
        for consent_type in CONSENT_TYPES:
            if consent_type in existing_types:
                continue
            db.add(
                ConsentRecord(
                    user_id=student.id,
                    consent_type=consent_type,
                    is_granted=True,
                    policy_version=CURRENT_POLICY_VERSION,
                    granted_at=datetime.utcnow(),
                    source="hosted_demo_bootstrap",
                )
            )
            created += 1
    return created


def main():
    parser = argparse.ArgumentParser(description="Create missing hosted demo master data without dropping existing data.")
    parser.add_argument("--grant-demo-consents", action="store_true", help="Grant all processing consents for existing demo students.")
    args = parser.parse_args()

    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        print("ADMIN_PASSWORD not set. Admin user will be created without changing its password.")

    db = SessionLocal()
    try:
        university, university_created = get_or_create_university(db)
        admin, admin_created = ensure_admin(db, admin_password)
        students_updated = ensure_student_university(db, university)
        profiles, profiles_created = ensure_counselor_profiles(db, university)
        university_assignments_created = ensure_university_assignments(db, profiles, university, admin)
        student_assignments_created = ensure_student_assignments(db, profiles, admin)
        consents_created = grant_demo_consents(db) if args.grant_demo_consents else 0

        db.commit()
        print("Hosted demo bootstrap complete")
        print(f"university_created={university_created}")
        print(f"admin_created={admin_created}")
        print(f"students_assigned_to_university={students_updated}")
        print(f"counselor_profiles_created={profiles_created}")
        print(f"counselor_university_assignments_created={university_assignments_created}")
        print(f"student_counselor_assignments_created={student_assignments_created}")
        print(f"demo_consents_created={consents_created}")
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()

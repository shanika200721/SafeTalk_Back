from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database_models import University, User, UserRole as DbUserRole
from app.schemas import Login, Token, User as UserSchema, UserCreate, UserLookup, UserUpdate
from app.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user from JWT token."""
    username = decode_token(token)
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.username == username).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )
    return user


@router.get("/universities")
def list_registration_universities(db: Session = Depends(get_db)):
    """List active universities for public student registration."""
    universities = (
        db.query(University)
        .filter(University.active.is_(True))
        .order_by(University.university_name.asc(), University.campus_name.asc())
        .all()
    )
    return {
        "universities": [
            {
                "id": university.id,
                "university_name": university.university_name,
                "university": university.university_name,
                "university_code": university.university_code,
                "campus_name": university.campus_name,
                "campus": university.campus_name,
                "district": university.district,
            }
            for university in universities
        ]
    }


@router.post("/register", response_model=UserSchema)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a public student account."""
    try:
        if user_data.role.value != "student":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Public registration is limited to student accounts. "
                    "Staff accounts require administrative provisioning."
                ),
            )

        existing_user = (
            db.query(User)
            .filter((User.email == user_data.email) | (User.username == user_data.username))
            .first()
        )

        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email or username already registered",
            )

        if user_data.university_id:
            university = (
                db.query(University)
                .filter(University.id == user_data.university_id, University.active.is_(True))
                .first()
            )
            if not university:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected university is not available",
                )

        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hash_password(user_data.password),
            role=DbUserRole.STUDENT,
            university_id=user_data.university_id,
            department=user_data.department,
            year_of_study=user_data.year_of_study,
        )

        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    except HTTPException:
        raise
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed",
        )


@router.post("/login", response_model=Token)
def login(credentials: Login, db: Session = Depends(get_db)):
    """Generic login for any user role."""
    return _login_user(credentials, db, allowed_roles=None)


@router.post("/login/student", response_model=Token)
def login_student(credentials: Login, db: Session = Depends(get_db)):
    """Login endpoint for students only."""
    return _login_user(credentials, db, allowed_roles=["student"])


@router.post("/login/counselor", response_model=Token)
def login_counselor(credentials: Login, db: Session = Depends(get_db)):
    """Login endpoint for counselor, psychiatrist, and admin staff."""
    return _login_user(credentials, db, allowed_roles=["counselor", "admin", "psychiatrist"])


def _login_user(credentials: Login, db: Session, allowed_roles=None):
    """Internal login logic."""
    login_identifier = credentials.username.strip()
    normalized_identifier = login_identifier.lower()
    user = (
        db.query(User)
        .filter(
            or_(
                func.lower(User.username) == normalized_identifier,
                func.lower(User.email) == normalized_identifier,
            )
        )
        .first()
    )

    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    if allowed_roles and user.role.value not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This login endpoint is not available for {user.role.value}s. Please use the correct login page.",
        )

    user.last_login_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires,
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role.value,
            "university_id": user.university_id,
            "department": user.department,
            "year_of_study": user.year_of_study,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
        },
    }


@router.get("/me", response_model=UserSchema)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return current_user


@router.put("/me", response_model=UserSchema)
def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update current user profile."""
    if user_update.full_name:
        current_user.full_name = user_update.full_name
    if user_update.department:
        current_user.department = user_update.department
    if user_update.year_of_study:
        current_user.year_of_study = user_update.year_of_study

    db.commit()
    db.refresh(current_user)
    return current_user


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)):
    """Logout user. The client should delete its token."""
    return {"message": "Successfully logged out"}


@router.get("/verify")
def verify_current_token(current_user: User = Depends(get_current_user)):
    """Verify the caller's bearer token without placing a token in the URL."""
    return {"valid": True, "user_id": current_user.id, "role": current_user.role.value}


@router.get("/users", response_model=list[UserLookup])
def get_all_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all users. Admin only."""
    if current_user.role != DbUserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view all users",
        )

    return db.query(User).all()


@router.get("/users/{user_id}", response_model=UserLookup)
def get_user_by_id(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get a restricted user profile by ID."""
    if user_id != current_user.id and current_user.role != DbUserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to view this user",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return user

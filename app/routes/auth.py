from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import timedelta
from app.database import get_db
from app.models.database_models import User
from app.schemas import UserCreate, UserUpdate, Token, User as UserSchema, Login
from app.security import hash_password, verify_password, create_access_token, decode_token, ACCESS_TOKEN_EXPIRE_MINUTES
from fastapi.security import OAuth2PasswordBearer

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current user from JWT token"""
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

@router.post("/register", response_model=UserSchema)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    
    try:
        print(f"📝 Registering user: {user_data.email}")
        
        # Check if user already exists
        existing_user = db.query(User).filter(
            (User.email == user_data.email) | (User.username == user_data.username)
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email or username already registered"
            )
        
        # Create new user
        print(f"🔐 Hashing password for {user_data.email}")
        db_user = User(
            email=user_data.email,
            username=user_data.username,
            full_name=user_data.full_name,
            hashed_password=hash_password(user_data.password),
            role=user_data.role,
            department=user_data.department,
            year_of_study=user_data.year_of_study
        )
        
        print(f"💾 Saving user to database")
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        print(f"✅ User registered successfully: {db_user.id}")
        return db_user
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {str(e)}")
        print(f"Error type: {type(e)}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Token)
def login(credentials: Login, db: Session = Depends(get_db)):
    """Generic login for any user role"""
    try:
        print(f"DEBUG: Login attempt for user: {credentials.username}")
        result = _login_user(credentials, db, allowed_roles=None)
        print(f"DEBUG: Login successful for user: {credentials.username}")
        return result
    except Exception as e:
        print(f"DEBUG: Login error for {credentials.username}: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@router.post("/login/student", response_model=Token)
def login_student(credentials: Login, db: Session = Depends(get_db)):
    """Login endpoint for students only"""
    return _login_user(credentials, db, allowed_roles=["student"])

@router.post("/login/counselor", response_model=Token)
def login_counselor(credentials: Login, db: Session = Depends(get_db)):
    """Login endpoint for counselors only"""
    return _login_user(credentials, db, allowed_roles=["counselor", "admin", "psychiatrist"])

def _login_user(credentials: Login, db: Session, allowed_roles=None):
    """Internal login logic"""
    
    # Find user by username
    user = db.query(User).filter(User.username == credentials.username).first()
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Check user role if restricted login
    if allowed_roles and user.role.value not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"This login endpoint is not available for {user.role.value}s. Please use the correct login page."
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},
        expires_delta=access_token_expires
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
            "department": user.department,
            "year_of_study": user.year_of_study,
            "is_active": user.is_active,
            "created_at": user.created_at,
            "updated_at": user.updated_at
        }
    }

@router.get("/me", response_model=UserSchema)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile"""
    return current_user

@router.put("/me", response_model=UserSchema)
def update_user_profile(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user profile"""
    
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
    """Logout user (client should delete token)"""
    return {"message": "Successfully logged out"}

@router.get("/verify/{token}")
def verify_token(token: str):
    """Verify if a token is valid"""
    username = decode_token(token)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False}

@router.get("/users", response_model=list)
def get_all_users(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all users (admin only)"""
    if current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can view all users"
        )
    
    users = db.query(User).all()
    return users

@router.get("/users/{user_id}", response_model=UserSchema)
def get_user_by_id(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return user

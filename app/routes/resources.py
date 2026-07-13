from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.database_models import Resource, User
from app.routes.auth import get_current_user
from pydantic import BaseModel

router = APIRouter(prefix="/api/resources", tags=["Resources"])

class ResourceCreate(BaseModel):
    title: str
    category: str
    description: str
    url: str = None
    phone: str = None

class ResourceResponse(BaseModel):
    id: int
    title: str
    category: str
    description: str
    url: str = None
    phone: str = None

@router.get("/")
def get_resources(
    category: str = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all available resources"""
    
    query = db.query(Resource).filter(Resource.is_active == True)
    
    if category:
        query = query.filter(Resource.category == category)
    
    resources = query.all()
    
    return {
        "total_resources": len(resources),
        "resources": [
            {
                "id": r.id,
                "title": r.title,
                "category": r.category,
                "description": r.description,
                "url": r.url,
                "phone": r.phone
            }
            for r in resources
        ]
    }

@router.get("/categories")
def get_resource_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all resource categories"""
    
    categories = db.query(Resource.category).filter(
        Resource.is_active == True
    ).distinct().all()
    
    return {
        "categories": [c[0] for c in categories]
    }

@router.get("/crisis")
def get_crisis_resources(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get crisis resources"""
    
    resources = db.query(Resource).filter(
        (Resource.category == "crisis") &
        (Resource.is_active == True)
    ).all()
    
    return {
        "crisis_resources": [
            {
                "id": r.id,
                "title": r.title,
                "description": r.description,
                "url": r.url,
                "phone": r.phone
            }
            for r in resources
        ]
    }

@router.post("/", response_model=dict)
def create_resource(
    resource_data: ResourceCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new resource (admin only)"""
    
    if current_user.role.value != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins can create resources"
        )
    
    db_resource = Resource(
        title=resource_data.title,
        category=resource_data.category,
        description=resource_data.description,
        url=resource_data.url,
        phone=resource_data.phone
    )
    
    db.add(db_resource)
    db.commit()
    db.refresh(db_resource)
    
    return {
        "id": db_resource.id,
        "title": db_resource.title,
        "category": db_resource.category,
        "message": "Resource created successfully"
    }

@router.get("/{resource_id}")
def get_resource(
    resource_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific resource"""
    
    resource = db.query(Resource).filter(Resource.id == resource_id).first()
    
    if not resource:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resource not found"
        )
    
    return {
        "id": resource.id,
        "title": resource.title,
        "category": resource.category,
        "description": resource.description,
        "url": resource.url,
        "phone": resource.phone
    }

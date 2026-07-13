# Suicide Prevention Agent - Backend Setup Guide

## Project Overview
This is a FastAPI-based backend for an AI-powered suicide prevention system that provides comprehensive mental health assessment, daily check-ins, and counselor support features.

## Features

### 1. **Authentication & User Management**
- User registration (Student, Counselor, Admin roles)
- JWT-based authentication
- Password hashing with bcrypt
- Token management

### 2. **Assessment Modules**
- **Profile Assessment**: Academic, family, and lifestyle factors
- **DASS21 Assessment**: Depression, Anxiety, and Stress evaluation
- **Daily Check-in**: Mood, sleep, exercise, stress tracking
- **Multimodal Risk Assessment**: Composite scoring from multiple modalities

### 3. **Risk Scoring System**
- Weighted multimodal assessment (7 modalities)
- Risk levels: LOW, MEDIUM, HIGH, SEVERE
- Automatic escalation for high-risk cases
- Personalized recommendations

### 4. **Counselor Features**
- High-risk user monitoring
- Counselor session management
- Student dashboard with comprehensive metrics
- Analytics and reporting
- Alert system for critical cases

### 5. **Resources Management**
- Crisis hotline information
- Support resources by category
- Mental health references

## Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager
- SQLite or PostgreSQL database

### Setup Steps

1. **Clone/Navigate to Backend Directory**
```bash
cd backend
```

2. **Create Virtual Environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# OR using conda
conda create -n suicide-prevention python=3.9
conda activate suicide-prevention
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
# Edit .env file with your configuration
# Default uses SQLite for development
# For production, use PostgreSQL
```

5. **Initialize Database**
```bash
# Database tables are created automatically on first run
python -m uvicorn app.main:app --reload
```

## Running the Server

### Development Mode
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Python Script
```bash
python run_server.py
```

The API will be available at: `http://localhost:8000`
- API Documentation: `http://localhost:8000/api/docs`
- Alternative Docs: `http://localhost:8000/api/redoc`

## API Endpoints

### Authentication (`/api/auth/`)
- `POST /register` - Register new user
- `POST /login` - Login user
- `GET /me` - Get current user profile
- `PUT /me` - Update user profile
- `GET /users/{user_id}` - Get user by ID
- `GET /verify/{token}` - Verify JWT token

### Assessments (`/api/assessments/`)
- `POST /profile` - Create profile assessment
- `GET /profile/{user_id}` - Get profile assessment
- `POST /dass21` - Create DASS21 assessment
- `GET /dass21/latest` - Get latest DASS21
- `GET /dass21/history` - Get DASS21 history
- `POST /risk-assessment` - Perform multimodal risk assessment
- `GET /history/{user_id}` - Get assessment history
- `GET /latest` - Get latest assessment

### Daily Check-in (`/api/checkin/`)
- `POST /today` - Create today's check-in
- `GET /today` - Get today's check-in status
- `GET /history` - Get check-in history
- `GET /stats` - Get check-in statistics

### Counselor (`/api/counselor/`)
- `GET /alerts` - Get high-risk alerts
- `PUT /alerts/{alert_id}/read` - Mark alert as read
- `GET /high-risk-users` - Get high-risk students
- `GET /student/{user_id}/dashboard` - Get student dashboard
- `POST /sessions` - Create counselor session
- `GET /sessions/{session_id}` - Get session details
- `PUT /sessions/{session_id}` - Update session
- `GET /analytics/summary` - Get analytics summary

### Resources (`/api/resources/`)
- `GET /` - Get all resources
- `GET /categories` - Get resource categories
- `GET /crisis` - Get crisis resources
- `POST /` - Create resource (admin only)
- `GET /{resource_id}` - Get specific resource

### Health & Info
- `GET /` - Root endpoint
- `GET /api/health` - Health check
- `GET /api/info` - API information

## Database Models

### Core Models
- **User**: Students, Counselors, Admins
- **ProfileAssessment**: Background information and risk factors
- **DailyCheckIn**: Daily mood and wellness tracking
- **DASS21Assessment**: Depression, Anxiety, Stress evaluation
- **Assessment**: Multimodal risk assessment results
- **CounselorSession**: Counselor-student interaction records
- **Alert**: System alerts for high-risk cases
- **Resource**: Crisis and support resources

## Risk Assessment Weights

The system uses weighted multimodal assessment:
- DASS21: 0.489 (48.9%)
- Text Analysis: 0.238 (23.8%)
- Behavioral: 0.109 (10.9%)
- Facial Analysis: 0.078 (7.8%)
- Voice Analysis: 0.050 (5.0%)
- Profile: 0.030 (3.0%)
- Mood: 0.006 (0.6%)

## Risk Levels

- **LOW** (< 30): Routine monitoring
- **MEDIUM** (30-60): Enhanced monitoring, counseling recommended
- **HIGH** (60-80): Urgent counselor notification
- **SEVERE** (> 80): Emergency escalation

## Security Features

- JWT token-based authentication
- Password hashing with bcrypt
- Role-based access control (RBAC)
- CORS protection
- HTTPS recommended for production
- Environment variable security

## Database Configuration

### Development (SQLite)
```
DATABASE_URL=sqlite:///./suicideprevention.db
```

### Production (PostgreSQL)
```
DATABASE_URL=postgresql://user:password@localhost:5432/suicide_prevention_db
```

## Environment Variables

Create a `.env` file with:

```
ENVIRONMENT=development
HOST=0.0.0.0
PORT=8000
DATABASE_URL=sqlite:///./suicideprevention.db
SECRET_KEY=your-super-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Frontend Integration

The frontend (React) connects to the backend at:
- Development: `http://localhost:8000`
- Production: Update the API base URL in frontend configuration

## Testing

### Test User Accounts

After starting the server, you can register users with different roles:

```bash
# Student Registration
POST /api/auth/register
{
    "email": "student@university.edu",
    "username": "student01",
    "full_name": "John Doe",
    "password": "SecurePass123!",
    "role": "student",
    "department": "Computer Science",
    "year_of_study": 2
}

# Counselor Registration
{
    "email": "counselor@university.edu",
    "username": "counselor01",
    "full_name": "Dr. Jane Smith",
    "password": "SecurePass123!",
    "role": "counselor"
}
```

### Test Assessment Flow

1. Create profile assessment
2. Submit DASS21 responses
3. Complete daily check-in
4. Perform multimodal risk assessment
5. View results and recommendations

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Use different port
   python -m uvicorn app.main:app --port 8001
   ```

2. **Database Error**
   ```bash
   # Delete existing database and restart
   rm suicideprevention.db
   python -m uvicorn app.main:app --reload
   ```

3. **CORS Error**
   - Update `ALLOWED_ORIGINS` in `.env`
   - Restart the server

4. **Module Import Error**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   ```

## Performance Optimization

- Use connection pooling for databases
- Implement caching for frequently accessed resources
- Use async endpoints for long-running operations
- Monitor API response times

## Deployment

### Docker Deployment
```bash
docker build -t suicide-prevention-api .
docker run -p 8000:8000 suicide-prevention-api
```

### Cloud Deployment (AWS, GCP, Azure)
- Deploy to managed container services (ECS, Cloud Run, Container Instances)
- Use managed databases (RDS, Cloud SQL, Azure SQL)
- Configure SSL/HTTPS certificates

## Support & Documentation

- API Docs: `http://localhost:8000/api/docs`
- Interactive API Testing: Swagger UI included
- Backend Structure: See `/app` directory
- Database Models: See `/app/models/database_models.py`

## License

This project is part of the Suicide Prevention Initiative.

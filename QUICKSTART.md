# 🚀 Quick Start Guide - Suicide Prevention Agent Backend

## 5-Minute Startup

### Step 1: Navigate to Backend
```bash
cd backend
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Server
```bash
python run_server.py
```

**You should see:**
```
🚀 Starting Suicide Prevention API...
📍 Environment: development
🌐 Host: 0.0.0.0
🔌 Port: 8000
🔄 Auto-reload: True

📚 API Documentation available at:
   • Swagger UI: http://localhost:8000/api/docs
   • ReDoc: http://localhost:8000/api/redoc
```

## Access the API

### Interactive API Documentation
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

### Health Check
```bash
curl http://localhost:8000/api/health
```

## Test the API (Using Swagger UI)

### 1. Register a Student
```bash
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
```

### 2. Login
```bash
POST /api/auth/login

{
  "username": "student01",
  "password": "SecurePass123!"
}
```

Copy the `access_token` from the response.

### 3. Create Profile Assessment
```bash
POST /api/assessments/profile

{
  "user_id": 1,
  "gpa": 3.5,
  "repeated_subjects": 1,
  "attendance": 95,
  "family_relationship_score": 8,
  "income_level": "Medium",
  "living_arrangement": "With Family",
  "communication_skills": 7,
  "social_connection": 7,
  "sleep_pattern": "Regular",
  "exercise_frequency": "Regularly",
  "substance_use": "None"
}
```

### 4. Submit DASS21 Assessment
```bash
POST /api/assessments/dass21

{
  "responses": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1]
}
```

### 5. Daily Check-in
```bash
POST /api/checkin/today

{
  "mood": 3,
  "sleep_hours": 7.5,
  "exercise_minutes": 30,
  "social_interaction": "Moderate",
  "stress_level": 6,
  "anxiety_level": 5,
  "negative_thoughts": false,
  "substance_use_today": false,
  "self_harm_thoughts": false
}
```

## Authentication

All endpoints (except registration and login) require:
```
Authorization: Bearer <your_access_token>
```

In Swagger UI: Click "Authorize" button and paste your token.

## Database

The database is automatically created on first run:
- **Development**: `suicideprevention.db` (SQLite)
- **Location**: `backend/suicideprevention.db`

## Common Commands

### Stop the Server
```
Ctrl + C (or Cmd + C on Mac)
```

### Deactivate Virtual Environment
```bash
deactivate
```

### Delete Database (Reset)
```bash
rm suicideprevention.db
# Then restart the server
```

### Use Different Port
Edit `.env` and change:
```
PORT=8001
```

## Configuration

Edit `.env` file to customize:
- `DATABASE_URL` - Change database
- `SECRET_KEY` - Security key
- `ALLOWED_ORIGINS` - CORS settings
- `PORT` - Server port

## Troubleshooting

### Port 8000 Already in Use
```bash
# Use a different port
python -m uvicorn app.main:app --port 8001
```

### Module Not Found Error
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### CORS Error from Frontend
Update in `.env`:
```
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # Main application
│   ├── database.py          # Database setup
│   ├── security.py          # JWT & passwords
│   ├── schemas.py           # Data models
│   ├── models/
│   │   ├── database_models.py   # Database tables
│   │   └── risk_assessment.py   # Risk engine
│   ├── routes/
│   │   ├── auth.py          # Authentication
│   │   ├── assessments.py   # Assessments
│   │   ├── checkin.py       # Daily check-in
│   │   ├── counselor.py     # Counselor features
│   │   └── resources.py     # Resources
│   └── utils/
│       ├── dass21_calculator.py     # DASS21 scoring
│       └── assessment_calculator.py # Risk calculations
├── requirements.txt         # Dependencies
├── .env                    # Configuration
└── run_server.py          # Server runner
```

## Key Features Available Now

✅ User registration and login  
✅ Profile assessment  
✅ DASS21 mental health evaluation  
✅ Daily mood tracking  
✅ Multimodal risk assessment  
✅ Counselor alerts and monitoring  
✅ Student dashboard  
✅ Crisis resources  

## Next: Frontend Integration

Connect your React frontend to:
```
http://localhost:8000
```

See `API_ENDPOINTS.md` for complete endpoint reference.

## Documentation Files

- **BACKEND_SETUP.md** - Complete setup guide
- **API_ENDPOINTS.md** - All endpoints with examples
- **PROJECT_SUMMARY.md** - Project overview
- **IMPLEMENTATION_CHECKLIST.md** - What's been built

## Support

For detailed information, see:
- API Docs: http://localhost:8000/api/docs
- Backend Setup: `BACKEND_SETUP.md`
- Endpoint Reference: `API_ENDPOINTS.md`

## Need Help?

1. Check if server is running: http://localhost:8000/api/health
2. Review `.env` configuration
3. Check `requirements.txt` is installed
4. Review error messages in terminal
5. See BACKEND_SETUP.md troubleshooting section

---

**Status**: ✅ Ready to Use!

Start the server and begin testing at http://localhost:8000/api/docs

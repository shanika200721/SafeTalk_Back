# SafeTalk Backend

Backend service for **SafeTalk — Multimodal AI Agent for Suicide Prevention in Sri Lankan Undergraduates**.

The backend provides secure APIs for user management, mental-health assessments, multimodal model inference, risk-score fusion, supportive interventions, and counselor-in-the-loop escalation. It is designed as an early-warning and decision-support platform and **not as a replacement for professional diagnosis, treatment, or emergency care**.

## Core Features

- Student and counselor authentication with role-based access
- Student profile and baseline-risk assessment
- DASS-21 scoring and severity classification
- Daily mood check-ins and longitudinal trend tracking
- Text, speech, and facial-emotion analysis endpoints
- Weighted late-fusion risk scoring with missing-modality handling
- Risk classification into low, medium, high, and severe levels
- SafeTalk supportive-chat history and safety controls
- Counselor alerts, case summaries, and escalation records
- Model, feature, prediction, and risk-assessment tracking
- PostgreSQL support with SQLite compatibility
- Alembic database migrations
- Automated API and backend tests

## Technology Stack

- **Language:** Python
- **API Framework:** FastAPI
- **Database:** PostgreSQL / SQLite
- **ORM:** SQLAlchemy
- **Migrations:** Alembic
- **Authentication:** JWT-based authentication
- **Machine Learning:** scikit-learn, pandas, NumPy
- **Text Processing:** TF-IDF and classical ML classifiers
- **Speech Processing:** Librosa-based acoustic features
- **Facial Analysis:** Image-feature and emotion-classification pipeline
- **Testing:** Pytest

## System Modules

### 1. Diagnosis Module

Collects and processes profile information, DASS-21 responses, mood records, text, speech, facial expressions, and other available signals.

### 2. Risk-Fusion Module

Normalizes available modality outputs and combines them using weighted late fusion. Missing modalities are excluded and the remaining weights are renormalized before calculating the final risk score.

### 3. Intervention and Escalation Module

Maps the calculated risk level to suitable supportive actions. High-risk and severe-risk cases can be surfaced to authorized counselors through a human-in-the-loop workflow.

## Project Structure

```text
backend/
├── alembic/                 # Database migration scripts
├── app/
│   ├── api/                 # API routes and request handlers
│   ├── core/                # Configuration, security, and shared services
│   ├── db/                  # Database session and persistence logic
│   ├── ml/                  # ML preprocessing, inference, and fusion modules
│   ├── models/              # SQLAlchemy database models
│   ├── schemas/             # Pydantic request/response schemas
│   ├── services/            # Business and application services
│   └── main.py              # FastAPI application entry point
├── tests/                   # Backend and integration tests
├── alembic.ini
├── requirements.txt
└── .env.example
```

> The exact folder names may differ slightly depending on the current repository version.

## Local Setup

### Prerequisites

- Python 3.10 or later
- PostgreSQL for the production-style setup
- Git

### 1. Clone the repository

```bash
git clone <BACKEND_REPOSITORY_URL>
cd <BACKEND_REPOSITORY_FOLDER>
```

### 2. Create and activate a virtual environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file using `.env.example` as the template.

```env
APP_NAME=SafeTalk Backend
APP_ENV=development
SECRET_KEY=replace-with-a-secure-secret
ACCESS_TOKEN_EXPIRE_MINUTES=60
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/safetalk
FRONTEND_ORIGIN=http://localhost:5173
MODEL_ROOT=./models
```

For a simple local SQLite setup:

```env
DATABASE_URL=sqlite:///./safetalk.db
```

Never commit real secrets, credentials, participant data, or production `.env` files.

### 5. Apply database migrations

```bash
alembic upgrade head
```

### 6. Start the API

```bash
uvicorn app.main:app --reload
```

The service will normally be available at:

- API: `http://127.0.0.1:8000`
- Swagger documentation: `http://127.0.0.1:8000/docs`
- ReDoc documentation: `http://127.0.0.1:8000/redoc`

## Testing

Run the complete backend test suite:

```bash
pytest
```

Run with coverage:

```bash
pytest --cov=app --cov-report=term-missing
```

## API Areas

Typical API groups include:

```text
/auth              Authentication and user sessions
/users             User and role management
/profile           Student profile assessments
/dass21            DASS-21 submissions and scoring
/checkins          Daily mood check-ins
/chat              SafeTalk conversations
/analysis/text     Text-risk analysis
/analysis/speech   Speech-emotion analysis
/analysis/face     Facial-emotion analysis
/risk              Fusion and risk assessments
/alerts            Counselor alerts and escalation
/health            Service-health verification
```



## Machine-Learning Workflow

```text
Raw Input
   ↓
Validation and Preprocessing
   ↓
Individual Modality Prediction
   ↓
Score Normalization
   ↓
Weighted Late Fusion
   ↓
Risk Level and Explanation
   ↓
Supportive Action / Counselor Review
```

The current research implementation includes classical baseline models and rule-based clinical scoring where appropriate. Model artifacts, preprocessing versions, feature schemas, and evaluation evidence should be versioned so that predictions remain reproducible and auditable.

## Data Protection and Safety

This repository handles highly sensitive mental-health information. Development and deployment must include:

- Informed and modality-specific consent
- Least-privilege access control
- Encrypted transport and secure credential management
- Data minimization and retention controls
- Audit logging for sensitive actions
- Human review for serious risk decisions
- Clear handling of false positives and false negatives
- No representation of the system as a clinical diagnosis tool


## Current Research Scope

SafeTalk is an undergraduate research system developed to evaluate the technical feasibility of multimodal mental-health risk assessment and counselor-assisted escalation. Outputs should be interpreted as supportive risk indicators requiring professional judgment, not as definitive clinical conclusions.


## Author

**M.A.S. Sewwandi**  
BSc (Hons) Computer Science  
Faculty of Computing and Technology, University of Kelaniya

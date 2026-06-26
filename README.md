# Heart Disease FastAPI + React App

This project turns the original notebook-style scripts into a full app:

- FastAPI backend APIs
- Neon PostgreSQL connection through Tortoise ORM
- Automatic table creation for training stats
- Supabase Storage upload for pickle artifacts
- React + Tailwind UI with two sidebar tabs
- CSV upload, model training, stats display, and patient prediction

## Structure

```text
heart disease/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI routes
│   │   ├── models.py        # Tortoise ORM TrainingRun table
│   │   ├── database.py      # DB init and schema creation
│   │   ├── ml_pipeline.py   # preprocessing, feature engineering, training, evaluation
│   │   ├── storage.py       # Supabase artifact upload/download
│   │   ├── schemas.py       # API request/response schemas
│   │   └── config.py        # environment settings
│   ├── artifacts/           # local artifact cache
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── main.jsx         # React UI
│   │   ├── api.js           # API client
│   │   └── index.css        # Tailwind styles
│   └── dist/                # production build for frontend hosting
└── heart.csv                # sample dataset
```

## Environment

Create `.env` from `.env.example`:

```bash
cd backend
cp .env.example .env
```

Fill these values:

```env
DATABASE_URL=postgres://USER:PASSWORD@HOST.neon.tech/DBNAME?ssl=true
SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_KEY=YOUR_SUPABASE_SERVICE_OR_ANON_KEY
SUPABASE_BUCKET=heart-model-artifacts
GROQ_API_KEY=YOUR_GROQ_API_KEY
GROQ_MODEL=llama-3.3-70b-versatile
```

Create the Supabase bucket named `heart-model-artifacts` before training.

## Backend

```bash
cd "/Users/osman/Downloads/heart disease/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

FastAPI will create the `training_runs` table automatically through Tortoise ORM.

## Frontend

For local React development:

```bash
cd "/Users/osman/Downloads/heart disease/frontend"
npm install
npm run dev
```

Open the Vite URL, usually:

```text
http://localhost:5173
```

The frontend Vite server proxies `/api/*` requests to `http://127.0.0.1:8000`.

For production frontend build:

```bash
cd "/Users/osman/Downloads/heart disease/frontend"
npm run build
```

The backend does not serve the React UI. Run the UI from the frontend dev server or deploy `frontend/dist` separately.

## API Endpoints

```text
POST /api/train
```

Uploads a CSV, runs preprocessing, feature engineering, model selection, evaluation, stores stats in Neon, and uploads pickle artifacts to Supabase.

```text
GET /api/runs
GET /api/runs/latest
GET /api/runs/{run_id}
```

Returns stored run stats for the UI.

```text
GET /api/runs/{run_id}/stats
GET /api/runs/{run_id}/stats/preprocessing
GET /api/runs/{run_id}/stats/feature-engineering
GET /api/runs/{run_id}/stats/model-selection
GET /api/runs/{run_id}/stats/evaluation
```

Returns typed pipeline stats for each training stage.

```text
POST /api/predict
```

Accepts patient input, downloads/caches model artifacts, applies the same preprocessing, stores the prediction in the database, and returns prediction probability plus label.

```text
GET /api/predictions
GET /api/predictions?run_id={run_id}
GET /api/predictions/{prediction_id}
DELETE /api/predictions/{prediction_id}
POST /api/predictions/{prediction_id}/analysis
POST /api/chat
```

Returns saved prediction history, deletes saved predictions, and uses Groq for prediction analysis/chatbot responses.

The frontend chatbot route is:

```text
/chatbot?prediction_id={prediction_id}
```

## Required CSV Columns

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach,
exang, oldpeak, slope, ca, thal, target
```

For prediction, the same columns are used except `target`.

# CLAUDE.md — Model Monitoring & Drift Detection System

## 🎯 Project Overview
A production-style MLOps monitoring system that detects when a deployed ML model's performance degrades due to data drift or concept drift. Built as a portfolio project demonstrating senior-level MLOps thinking.

## 🧰 Tech Stack
- **Language:** Python 3.10+
- **ML:** scikit-learn (LogisticRegression or RandomForest for baseline)
- **Monitoring:** Evidently AI (drift detection, data quality reports)
- **Tracking:** MLflow (experiment tracking, model registry, drift metric logging)
- **API:** FastAPI + Uvicorn (prediction endpoint, drift status endpoint, health check)
- **Dashboard:** Streamlit + Plotly (live monitoring UI with alerts)
- **Dataset:** UCI Bank Marketing Dataset (binary classification — client subscribes to term deposit: yes/no)

## 📂 Project Structure
```
model-monitoring-system/
├── data/
│   ├── raw/                  # original dataset
│   ├── reference/            # baseline data for drift comparison
│   └── production/           # simulated production batches (drifted data)
├── src/
│   ├── train.py              # baseline model training + MLflow logging
│   ├── drift_simulator.py    # generates drifted production data batches
│   ├── monitor.py            # Evidently drift detection + report generation
│   └── utils.py              # shared helpers (data loading, preprocessing)
├── api/
│   └── main.py               # FastAPI app (predict, drift status, health)
├── dashboard/
│   └── app.py                # Streamlit monitoring dashboard
├── reports/                  # Evidently HTML/JSON drift reports
├── mlruns/                   # MLflow tracking directory
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## 📊 Dataset Details
- **Source:** UCI Bank Marketing Dataset (ID: 222)
- **Task:** Binary classification (predict if client subscribes to term deposit)
- **Features:** Mix of numerical (age, balance, duration, campaign) and categorical (job, marital, education, contact, month, poutcome)
- **Target:** `y` (yes/no)
- **Loading method:** `ucimlrepo` package → `fetch_ucirepo(id=222)`

## 🏗️ Build Phases
1. **Phase 1 — Foundation & Setup** ✅ (done)
2. **Phase 2 — Baseline Model Training + MLflow Integration** ← CURRENT
3. **Phase 3 — Drift Simulation Engine**
4. **Phase 4 — Evidently AI Monitoring**
5. **Phase 5 — FastAPI Backend**
6. **Phase 6 — Streamlit Dashboard**
7. **Phase 7 — Automation & Alerting**

## ⚙️ Coding Conventions
- Use type hints in all function signatures
- Add docstrings to all functions (Google style)
- Use pathlib for file paths, not os.path
- Constants in UPPER_CASE at module top
- Keep functions focused — one responsibility per function
- Use logging module, not print statements
- Save all Evidently reports as both HTML and JSON
- Log all metrics and artifacts to MLflow
- Error handling with try/except for I/O and model operations

## 🔑 Key Design Decisions
- **Reference vs Production split:** Training data serves as reference. Simulated production batches are compared against reference for drift.
- **Drift types to detect:** Data drift (feature distribution shift), concept drift (model performance degradation), target drift (label distribution change)
- **Drift simulation strategies:** Gradual drift (slow distribution shift over batches), sudden drift (abrupt distribution change), feature-level drift (only specific features shift)
- **MLflow usage:** Track training experiments AND drift detection results as separate experiment runs
- **Monitoring granularity:** Batch-level monitoring (not real-time streaming)

## ❌ What NOT to Do
- Do not use Jupyter notebooks — all code in .py files
- Do not hardcode file paths — use constants or config
- Do not skip error handling on data loading or model operations
- Do not use print for logging — use Python logging module
- Do not install unnecessary dependencies
- Do not create overly complex model — baseline simplicity is the point, monitoring is the star
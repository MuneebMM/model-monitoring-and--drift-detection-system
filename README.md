# Model Monitoring & Drift Detection System

A production-style MLOps monitoring system that detects when a deployed ML model's performance degrades due to data drift or concept drift. Built with scikit-learn, Evidently AI, MLflow, FastAPI, and Streamlit.

The system trains a baseline classifier on the UCI Bank Marketing dataset, simulates realistic production data drift scenarios, detects drift using statistical tests, serves predictions via a REST API, and visualizes everything through an interactive dashboard.

---

## Architecture

```
                    ┌──────────────┐
                    │  UCI Dataset │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  train.py    │  Baseline model + MLflow logging
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │   drift_simulator.py    │  Generates 10 production batches
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │     monitor.py          │  Evidently drift reports + MLflow
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌─────▼──────┐
   │  FastAPI   │   │  Streamlit  │   │  scheduler  │
   │  (api/)    │   │  (dashboard)│   │  (alerting) │
   └───────────┘   └─────────────┘   └────────────┘
```

## Project Structure

```
.
├── src/
│   ├── utils.py              # Data loading, preprocessing, Preprocessor class
│   ├── train.py              # Baseline RandomForest training + MLflow
│   ├── drift_simulator.py    # Generates drifted production batches
│   ├── monitor.py            # Evidently drift detection + report generation
│   └── scheduler.py          # Automated monitoring + alerting system
├── api/
│   └── main.py               # FastAPI prediction & monitoring endpoints
├── dashboard/
│   └── app.py                # Streamlit interactive monitoring dashboard
├── data/
│   ├── raw/                  # Original dataset + test split
│   ├── reference/            # Baseline data for drift comparison
│   └── production/           # Simulated production batches
├── models/                   # Trained model + preprocessor pickles
├── reports/                  # Evidently HTML/JSON reports + drift summary
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Dataset

**UCI Bank Marketing Dataset** (ID: 222) — Binary classification predicting whether a client will subscribe to a term deposit.

- **45,211 rows**, 16 features (7 numerical, 9 categorical)
- **Target:** `y` (yes/no) — heavily imbalanced (~11.7% positive class)
- **Key features:** age, job, balance, duration, campaign, contact method, previous outcome

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | scikit-learn (RandomForestClassifier) |
| Drift Detection | Evidently AI (DataDriftPreset, ValueDrift, ClassificationPreset) |
| Experiment Tracking | MLflow (SQLite backend) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Data Loading | ucimlrepo |

## Setup

```bash
# Clone the repository
git clone https://github.com/MuneebMM/Model-Monitoring---Drift-Detection-System.git
cd "Model Monitoring & Drift Detection System"

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Train the Baseline Model

```bash
python src/train.py
```

Trains a RandomForestClassifier (max_depth=10, 100 estimators), evaluates on a held-out test set, saves the model to `models/`, and logs everything to MLflow.

**Baseline metrics:**

| Metric | Score |
|---|---|
| Accuracy | 0.9026 |
| Precision | 0.6699 |
| Recall | 0.3299 |
| F1 | 0.4421 |
| ROC AUC | 0.9201 |

### 2. Simulate Production Drift

```bash
python src/drift_simulator.py
```

Generates 10 production batches (1,000 rows each) with controlled drift:

| Batches | Drift Type | What Changes |
|---|---|---|
| 01-02 | No drift | Minor random noise only |
| 03-07 | Gradual drift | Age, balance, duration, campaign shift progressively |
| 08 | Sudden drift | Abrupt demographic shift — older, wealthier, retired |
| 09-10 | Concept drift | Features stable, but label relationship changes |

### 3. Run Drift Monitoring

```bash
python src/monitor.py
```

Runs Evidently AI drift detection on all production batches against the reference data. For each batch:
- Generates HTML + JSON drift reports
- Evaluates model performance (accuracy, precision, recall, F1)
- Logs all metrics to MLflow as child runs
- Saves a summary CSV to `reports/drift_summary.csv`

### 4. Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Redirects to Swagger docs |
| GET | `/health` | Model status + latest drift info |
| POST | `/predict` | Single-row prediction (JSON input) |
| GET | `/drift/summary` | Full drift summary across all batches |
| GET | `/drift/batch/{n}` | Detailed drift report for batch n |
| POST | `/monitor/run` | Upload CSV for ad-hoc drift detection |

**Example prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 45, "job": "management", "marital": "married",
    "education": "tertiary", "default": "no", "balance": 5000,
    "housing": "no", "loan": "no", "contact": "cellular",
    "day_of_week": 20, "month": "mar", "duration": 800,
    "campaign": 1, "pdays": 100, "previous": 3, "poutcome": "success"
  }'
```

```json
{"prediction": "yes", "probability": 0.8379}
```

### 5. Launch the Dashboard

```bash
# Make sure the API is running on port 8000 first
streamlit run dashboard/app.py
```

The dashboard has four pages:
- **Overview** — KPI cards, performance trend lines, drift bar chart, alert banners
- **Drift Analysis** — Feature drift heatmap across batches, filterable summary table
- **Batch Explorer** — Per-batch deep dive with distribution plots and embedded Evidently reports
- **Live Prediction** — Interactive form that calls the prediction API with a probability gauge

### 6. Automated Monitoring & Alerting

```bash
# One-shot: process all unprocessed batches
python src/scheduler.py --mode once

# Continuous: poll for new batches every 30 seconds
python src/scheduler.py --mode watch
```

The scheduler watches `data/production/` for new batch files, runs drift detection automatically, logs to MLflow, updates the summary CSV, and writes alerts to `reports/alerts.log`.

**Alert thresholds:**

| Level | Condition |
|---|---|
| WARNING | Drifted feature share > 15% |
| WARNING | Target drift detected (p-value < 0.05) |
| CRITICAL | Accuracy drops > 5% below baseline |

## Drift Detection Results

| Batch | Drift Type | Drifted Share | Accuracy | F1 | Alert |
|---|---|---|---|---|---|
| batch_01 | no_drift | 0.0% | 0.930 | 0.573 | - |
| batch_02 | no_drift | 0.0% | 0.926 | 0.580 | - |
| batch_03 | gradual_drift | 22.2% | 0.928 | 0.640 | WARNING |
| batch_04 | gradual_drift | 22.2% | 0.924 | 0.558 | WARNING |
| batch_05 | gradual_drift | 22.2% | 0.923 | 0.570 | WARNING |
| batch_06 | gradual_drift | 22.2% | 0.899 | 0.530 | WARNING |
| batch_07 | gradual_drift | 22.2% | 0.893 | 0.552 | WARNING |
| batch_08 | sudden_drift | 22.2% | 0.940 | 0.700 | WARNING |
| batch_09 | concept_drift | 5.6% | 0.792 | 0.358 | CRITICAL |
| batch_10 | concept_drift | 5.6% | 0.794 | 0.376 | CRITICAL |

**Key findings:**
- **Gradual drift** (batches 03-07): Accuracy degrades steadily from 93% to 89% as feature distributions shift
- **Sudden drift** (batch 08): High accuracy (94%) despite severe feature drift — the shifted demographic is easier to classify, masking the problem
- **Concept drift** (batches 09-10): Lowest feature drift (5.6%) but worst accuracy (79%) — the model's decision boundary is wrong even though inputs look normal. This is the most dangerous drift type

## Screenshots

### MLflow Experiment Tracking

![MLflow Run Overview](Screenshots/Screenshot%20from%202026-04-08%2020-28-50.png)

![MLflow Metrics and Parameters](Screenshots/Screenshot%20from%202026-04-08%2020-29-11.png)

### Evidently Drift Reports

![Dataset Drift Summary](Screenshots/Screenshot%20from%202026-04-08%2020-50-06.png)

![Target Drift Analysis](Screenshots/Screenshot%20from%202026-04-08%2020-50-30.png)

![Target Drift Percentage View](Screenshots/Screenshot%20from%202026-04-08%2020-50-38.png)

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to explore experiment runs, compare metrics across batches, and inspect logged artifacts.

## How It All Connects

1. **`train.py`** produces a baseline model and saves the training data as the drift reference
2. **`drift_simulator.py`** generates production batches with known drift patterns
3. **`monitor.py`** compares each batch against the reference using Evidently statistical tests
4. **`scheduler.py`** automates step 3 — watches for new batches, runs detection, fires alerts
5. **`api/main.py`** serves predictions and exposes drift results via REST endpoints
6. **`dashboard/app.py`** visualizes drift trends, lets you explore individual batches, and test live predictions
7. **MLflow** ties everything together — every training run and monitoring result is tracked with full lineage

## License

This project is for educational and portfolio purposes.

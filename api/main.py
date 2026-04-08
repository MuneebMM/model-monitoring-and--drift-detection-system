"""FastAPI backend for model predictions, health checks, and drift monitoring."""

import io
import json
import logging
import pickle
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

# Add src/ to path so we can import monitor helpers
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitor import (
    CATEGORICAL_COLS,
    DRIFT_P_VALUE_THRESHOLD,
    DRIFT_SHARE_THRESHOLD,
    NUMERICAL_COLS,
    TARGET_COL,
    add_predictions,
    build_datasets,
    run_drift_report,
)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = PROJECT_ROOT / "models" / "baseline_model.pkl"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.pkl"
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference_data.csv"
DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift_summary.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App state (populated at startup) ──────────────────────────────────────────
_state: Dict[str, Any] = {}


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PredictionInput(BaseModel):
    """Single-row feature input matching the Bank Marketing dataset schema."""

    age: int = Field(..., ge=0, le=120, description="Client age in years")
    job: str = Field(..., description="Type of job")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default?")
    balance: int = Field(..., description="Average yearly balance in euros")
    housing: str = Field(..., description="Has housing loan?")
    loan: str = Field(..., description="Has personal loan?")
    contact: str = Field(..., description="Contact communication type")
    day_of_week: int = Field(..., ge=1, le=31, description="Last contact day of month")
    month: str = Field(..., description="Last contact month of year")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Contacts performed during campaign")
    pdays: int = Field(..., description="Days since previous campaign contact (-1 if none)")
    previous: int = Field(..., ge=0, description="Contacts performed before this campaign")
    poutcome: str = Field(..., description="Outcome of previous marketing campaign")


class PredictionResponse(BaseModel):
    """Prediction result with probability."""

    prediction: str = Field(..., description="Predicted class: 'yes' or 'no'")
    probability: float = Field(..., description="Probability of 'yes' class")


class HealthResponse(BaseModel):
    """API and model health status."""

    status: str
    model_loaded: bool
    reference_data_rows: int
    last_drift_check: Optional[str]
    latest_batch: Optional[str]
    latest_batch_drift_detected: Optional[bool]
    latest_batch_accuracy: Optional[float]


class BatchDriftDetail(BaseModel):
    """Detailed drift info for a single batch."""

    batch_name: str
    drift_type: Optional[str] = None
    dataset_drift_detected: Optional[bool] = None
    drifted_feature_share: Optional[float] = None
    target_drift_detected: Optional[bool] = None
    target_drift_p_value: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    drift_report: Optional[Dict[str, Any]] = None
    classification_report: Optional[Dict[str, Any]] = None


class MonitorRunResponse(BaseModel):
    """Result of an ad-hoc monitoring run on uploaded data."""

    rows_received: int
    dataset_drift_detected: bool
    drifted_feature_share: float
    target_drift_detected: Optional[bool] = None
    target_drift_p_value: Optional[float] = None
    feature_drift_scores: Dict[str, float]
    report_saved: str


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts and reference data on startup."""
    logger.info("Loading model artifacts...")
    try:
        with open(MODEL_PATH, "rb") as f:
            _state["model"] = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            _state["preprocessor"] = pickle.load(f)
        logger.info("Model and preprocessor loaded")
    except FileNotFoundError as exc:
        logger.error("Model artifacts missing: %s", exc)
        raise

    try:
        _state["reference"] = pd.read_csv(REFERENCE_PATH)
        logger.info("Reference data loaded: %d rows", len(_state["reference"]))
    except FileNotFoundError as exc:
        logger.error("Reference data missing: %s", exc)
        raise

    # Pre-compute reference predictions for drift comparison
    _state["reference_with_preds"] = add_predictions(
        _state["reference"], _state["model"], _state["preprocessor"]
    )

    _state["drift_summary"] = _load_drift_summary()
    _state["startup_time"] = datetime.now(timezone.utc).isoformat()

    logger.info("API ready")
    yield
    logger.info("Shutting down")


def _load_drift_summary() -> Optional[pd.DataFrame]:
    """Load drift summary CSV if it exists.

    Returns:
        DataFrame or None.
    """
    if DRIFT_SUMMARY_PATH.exists():
        df = pd.read_csv(DRIFT_SUMMARY_PATH)
        logger.info("Drift summary loaded: %d batches", len(df))
        return df
    logger.warning("No drift summary found at %s", DRIFT_SUMMARY_PATH)
    return None


app = FastAPI(
    title="Bank Marketing Model Monitor",
    description="Prediction serving, drift detection, and monitoring API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: PredictionInput) -> PredictionResponse:
    """Generate a prediction for a single customer.

    Args:
        input_data: Customer features matching the Bank Marketing schema.

    Returns:
        Prediction (yes/no) and probability of subscribing.
    """
    row = pd.DataFrame([input_data.model_dump()])

    preprocessor = _state["preprocessor"]
    for col in CATEGORICAL_COLS:
        row[col] = row[col].astype(str)
        # Map unseen labels to "nan" (the missing-value token in the encoder)
        known = set(preprocessor.label_encoders[col].classes_)
        row[col] = row[col].where(row[col].isin(known), other="nan")
    for col in NUMERICAL_COLS:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    encoded = preprocessor.transform(row)
    pred_class = int(_state["model"].predict(encoded)[0])
    pred_proba = float(_state["model"].predict_proba(encoded)[0][1])

    return PredictionResponse(
        prediction="yes" if pred_class == 1 else "no",
        probability=round(pred_proba, 4),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return API health status, model state, and latest drift info."""
    summary = _state.get("drift_summary")

    last_drift_check: Optional[str] = None
    latest_batch: Optional[str] = None
    drift_detected: Optional[bool] = None
    latest_accuracy: Optional[float] = None

    if summary is not None and not summary.empty:
        last_row = summary.iloc[-1]
        latest_batch = str(last_row["batch_name"])
        drift_detected = bool(last_row["dataset_drift_detected"])
        latest_accuracy = float(last_row["accuracy"])
        # Use file modification time of the summary as last check time
        mtime = DRIFT_SUMMARY_PATH.stat().st_mtime
        last_drift_check = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()

    return HealthResponse(
        status="healthy",
        model_loaded=_state.get("model") is not None,
        reference_data_rows=len(_state.get("reference", [])),
        last_drift_check=last_drift_check,
        latest_batch=latest_batch,
        latest_batch_drift_detected=drift_detected,
        latest_batch_accuracy=latest_accuracy,
    )


@app.get("/drift/summary")
async def drift_summary() -> List[Dict[str, Any]]:
    """Return the full drift summary across all monitored batches.

    Returns:
        List of per-batch drift metrics.

    Raises:
        HTTPException: 404 if no drift summary exists.
    """
    summary = _state.get("drift_summary")
    if summary is None or summary.empty:
        raise HTTPException(status_code=404, detail="No drift summary available. Run src/monitor.py first.")
    return summary.to_dict(orient="records")


@app.get("/drift/batch/{batch_number}", response_model=BatchDriftDetail)
async def drift_batch_detail(batch_number: int) -> BatchDriftDetail:
    """Return detailed drift info for a specific batch.

    Reads summary metrics from drift_summary.csv and the full Evidently
    JSON reports from reports/.

    Args:
        batch_number: Batch number (1-based, e.g. 1 for batch_01).

    Raises:
        HTTPException: 404 if the batch does not exist.
    """
    batch_name = f"batch_{batch_number:02d}"

    # Get summary row
    summary = _state.get("drift_summary")
    summary_row: Dict[str, Any] = {}
    if summary is not None and not summary.empty:
        match = summary[summary["batch_name"] == batch_name]
        if not match.empty:
            summary_row = match.iloc[0].to_dict()

    # Read JSON reports
    drift_json_path = REPORTS_DIR / f"{batch_name}_drift_report.json"
    cls_json_path = REPORTS_DIR / f"{batch_name}_classification_report.json"

    if not drift_json_path.exists() and not summary_row:
        raise HTTPException(status_code=404, detail=f"Batch {batch_name} not found")

    drift_report = None
    if drift_json_path.exists():
        try:
            with open(drift_json_path) as f:
                drift_report = json.load(f)
        except Exception as exc:
            logger.error("Failed to read %s: %s", drift_json_path, exc)

    cls_report = None
    if cls_json_path.exists():
        try:
            with open(cls_json_path) as f:
                cls_report = json.load(f)
        except Exception as exc:
            logger.error("Failed to read %s: %s", cls_json_path, exc)

    return BatchDriftDetail(
        batch_name=batch_name,
        drift_type=summary_row.get("drift_type"),
        dataset_drift_detected=summary_row.get("dataset_drift_detected"),
        drifted_feature_share=summary_row.get("drifted_feature_share"),
        target_drift_detected=summary_row.get("target_drift_detected"),
        target_drift_p_value=summary_row.get("target_drift_p_value"),
        accuracy=summary_row.get("accuracy"),
        precision=summary_row.get("precision"),
        recall=summary_row.get("recall"),
        f1=summary_row.get("f1"),
        drift_report=drift_report,
        classification_report=cls_report,
    )


@app.post("/monitor/run", response_model=MonitorRunResponse)
async def monitor_run(file: UploadFile = File(...)) -> MonitorRunResponse:
    """Run ad-hoc drift detection on uploaded production data.

    Accepts a CSV file, runs Evidently DataDriftPreset + target drift
    against the reference data, saves reports, and returns results.

    Args:
        file: Uploaded CSV file with production data.

    Raises:
        HTTPException: 422 if the CSV cannot be parsed or is missing columns.
    """
    try:
        contents = await file.read()
        current = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV: {exc}")

    required_features = set(NUMERICAL_COLS + CATEGORICAL_COLS)
    missing = required_features - set(current.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"CSV is missing required columns: {sorted(missing)}",
        )

    has_target = TARGET_COL in current.columns
    if not has_target:
        current[TARGET_COL] = 0

    try:
        current = add_predictions(
            current, _state["model"], _state["preprocessor"]
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {exc}")

    reference_with_preds = _state["reference_with_preds"]
    ref_ds, cur_ds = build_datasets(reference_with_preds, current)

    snap, drift_metrics = run_drift_report(ref_ds, cur_ds, "adhoc_upload")

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_name = f"adhoc_{timestamp}"
    html_path = REPORTS_DIR / f"{report_name}_drift_report.html"
    json_path = REPORTS_DIR / f"{report_name}_drift_report.json"
    try:
        snap.save_html(str(html_path))
        snap.save_json(str(json_path))
    except OSError as exc:
        logger.error("Failed to save adhoc reports: %s", exc)

    return MonitorRunResponse(
        rows_received=len(current),
        dataset_drift_detected=drift_metrics.get("dataset_drift_detected", False),
        drifted_feature_share=round(drift_metrics.get("drifted_feature_share", 0.0), 4),
        target_drift_detected=drift_metrics.get("target_drift_detected") if has_target else None,
        target_drift_p_value=drift_metrics.get("target_drift_p_value") if has_target else None,
        feature_drift_scores={
            k: round(v, 6) for k, v in drift_metrics.get("feature_drift_scores", {}).items()
        },
        report_saved=str(html_path),
    )
